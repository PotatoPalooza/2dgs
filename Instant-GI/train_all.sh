#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/work/10958/soumyabrata/ls6/2dgs/mipnerf360"
CONFIG_TEMPLATE="/work/10958/soumyabrata/ls6/2dgs/Instant-GI/datasets/general.yaml"

for SCENE_DIR in "$DATA_ROOT"/*/; do
    SCENE_NAME=$(basename "$SCENE_DIR")
    echo "Processing scene: $SCENE_NAME"

    # Find all JPG images
    IMAGE_LIST=()
    while IFS= read -r -d $'\0' file; do
        IMAGE_LIST+=("$file")
    done < <(find "$SCENE_DIR/images" -type f -iname "*.jpg" -print0)

    if [ ${#IMAGE_LIST[@]} -eq 0 ]; then
        echo "No images found in $SCENE_DIR/images, skipping..."
        continue
    fi

    # Convert to YAML list format
    IMAGES_YAML=$(printf '  - "%s"\n' "${IMAGE_LIST[@]}")

    # Create temp YAML
    SCENE_YAML="./datasets/temp_${SCENE_NAME}.yaml"

    # Replace or add dataset_name and image_paths
    awk -v scene="$SCENE_NAME" -v images_yaml="$IMAGES_YAML" '
        BEGIN {dataset_block=0}
        /^dataset:/ {print; dataset_block=1; next}
        dataset_block && /^  dataset_name:/ {print "  dataset_name: " scene; next}
        dataset_block && /^  image_paths:/ {
            print "  image_paths:\n" images_yaml
            dataset_block=0
            next
        }
        {print}
    ' "$CONFIG_TEMPLATE" > "$SCENE_YAML"

    python train.py --config "$SCENE_YAML"
done
