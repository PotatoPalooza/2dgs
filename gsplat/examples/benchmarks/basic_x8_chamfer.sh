SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark_8x_chamfer01"
# garden bicycle stump bonsai counter kitchen room
SCENE_LIST="garden bicycle stump bonsai counter kitchen room" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi
    DATA_FACTOR=8

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default --eval_steps -1 --disable_viewer --data_factor $DATA_FACTOR \
        --render_traj_path $RENDER_TRAJ_PATH \
        --data_dir data/360_v2/$SCENE/ \
        --result_dir $RESULT_DIR/$SCENE/ \
        --gaussian_data_dir data/360_v2_gs/{$SCENE}_d8/ \
        --chamfer_loss \
        --chamfer_lambda 0.02

    # run eval and render
    for CKPT in $RESULT_DIR/$SCENE/ckpts/*;
    do
        CUDA_VISIBLE_DEVICES=2 python simple_trainer.py default --disable_viewer --data_factor $DATA_FACTOR \
            --render_traj_path $RENDER_TRAJ_PATH \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --ckpt $CKPT \
            --gaussian_data_dir data/360_v2_gs/{$SCENE}_d8/ \
            --chamfer_loss \
            --chamfer_lambda 0.02
    done
done


for SCENE in $SCENE_LIST;
do
    echo "=== Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/val*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done