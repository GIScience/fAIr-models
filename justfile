set shell := ["bash", "-euo", "pipefail", "-c"]

compose := "docker compose -f infra/compose/docker-compose.yml"

[doc('Install deps, bring up the stack, register the ZenML stack')]
setup:
    uv sync --group dev --group docs --extra k8s
    uv run pre-commit install --hook-type commit-msg --hook-type pre-commit
    {{ compose }} up -d --wait
    uv run zenml init >/dev/null
    infra/compose/register-stack.sh
    @echo
    @echo "Stack up. ZenML :8080  MLflow :5000  STAC :8082  MinIO :9001"
    @echo "Next: 'just build' to build model images, then 'just example'."

[doc('Build model image(s). No arg = all; arg = one (e.g. `just build unet_segmentation`)')]
build model="":
    #!/usr/bin/env bash
    set -euo pipefail
    dirs=$([ -n "{{ model }}" ] && echo "models/{{ model }}" || echo models/*)
    for d in $dirs; do
        [[ -f "$d/Dockerfile" && -f "$d/stac-item.json" ]] || continue
        name=$(basename "$d")
        href=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['assets']['mlm:training']['href'])" "$d/stac-item.json")
        echo "==> building $name -> $href"
        docker build -f "$d/Dockerfile" --target runtime -t "$href" .
    done

[doc('Run a model inference container locally (e.g. `just serve unet_segmentation`)')]
serve model:
    #!/usr/bin/env bash
    set -euo pipefail
    href=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['assets']['mlm:inference']['href'])" "models/{{ model }}/stac-item.json")
    docker build -f "models/{{ model }}/Dockerfile" --target inference -t "$href" .
    echo "Serving on http://localhost:8080 (Ctrl-C to stop)"
    docker run --rm -it --network host "$href"

[doc('Run example pipeline(s). No arg = all; arg = one (e.g. `just example segmentation`)')]
example name="":
    #!/usr/bin/env bash
    set -euo pipefail
    exs=$([ -n "{{ name }}" ] && echo "{{ name }}" || echo "segmentation classification detection")
    for ex in $exs; do
        AWS_ENDPOINT_URL=http://localhost:9000 \
        AWS_ACCESS_KEY_ID=minioadmin \
        AWS_SECRET_ACCESS_KEY=minioadmin \
        FAIR_STAC_API_URL=http://localhost:8082 \
        FAIR_DSN=postgresql://postgres:postgres@localhost:5432/fair_models \
        FAIR_UPLOAD_ARTIFACTS=true \
            uv run python "examples/$ex/run.py"
    done

[doc('Stop the stack (containers stopped, volumes and ZenML state preserved)')]
down:
    {{ compose }} stop

[doc('Bring the stack back up after `just down`')]
up:
    {{ compose }} start

[doc('Destroy the stack: containers + volumes + local ZenML state + artifacts')]
tear:
    -{{ compose }} down -v
    -uv run zenml clean -y
    rm -rf .zen artifacts dist *.egg-info

[doc('Lint and format')]
lint:
    uv run ruff check --fix . && uv run ruff format . && uv run ty check

[doc('Run tests')]
test:
    uv run pytest -v

[doc('Validate STAC items and model pipelines')]
validate:
    uv run python scripts/validate_stac_items.py && uv run python scripts/validate_model.py

[doc('Serve documentation locally')]
docs:
    uv sync --group docs && uv run zensical serve

[doc('Run pre-commit hooks and commitizen')]
commit:
    uv run pre-commit run --all-files && uv run cz commit
