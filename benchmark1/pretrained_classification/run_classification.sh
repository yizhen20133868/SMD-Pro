export PYTHONPATH=../..
SEED=11111
BATCH_SIZE=16
EPOCHS=500
THRESHOLD=0.5
MODEL_NAME=$1
SAVED_NAME_PREFIX=classification

if [ ${MODEL_NAME} = bert ]
then
  LEARNING_RATE=5e-05
elif [ ${MODEL_NAME} = roberta ]
then
  LEARNING_RATE=1e-05
elif [ ${MODEL_NAME} = albert ]
then
  LEARNING_RATE=2e-05
elif [ ${MODEL_NAME} = electra ]
then
  LEARNING_RATE=1e-06
fi

SAVE_MODEL_NAME=${SAVED_NAME_PREFIX}_batch${BATCH_SIZE}_lr${LEARNING_RATE}_seed${SEED}_ptm${MODEL_NAME}

echo start training ${MODEL_NAME}
python run_classification.py \
    --seed=${SEED} \
    --batch_size=${BATCH_SIZE} \
    --lr=${LEARNING_RATE} \
    --epochs=${EPOCHS} \
    --model_name=${MODEL_NAME} \
    --saved_model_name=${SAVE_MODEL_NAME} \
    --threshold=${THRESHOLD}
echo start predicting ${MODEL_NAME}
python run_classification.py \
    --seed=${SEED} \
    --batch_size=${BATCH_SIZE} \
    --lr=${LEARNING_RATE} \
    --epochs=${EPOCHS} \
    --model_name=${MODEL_NAME} \
    --saved_model_name=${SAVE_MODEL_NAME} \
    --load_weights=True \
    --just_eval=True \
    --threshold=${THRESHOLD}