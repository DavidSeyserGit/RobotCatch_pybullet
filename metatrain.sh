#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
GENERATIONS=5
POPULATION=5
# 50 episodes * max 500 steps each
TIMESTEPS=$((50 * 500))
EVAL_EPISODES=3
TRAIN_SCRIPT="reinforcement_learning/sac+her_training.py"
REFINEMENT_SCRIPT="reinforcement_learning/ppo_refinement.py"
MODEL_NAME="reinforcement_learning/runs/run1/her_sac_robot.zip"
# --------------

best_model=""

for gen in $(seq 1 $GENERATIONS); do
    echo
    echo "===== Generation $gen ====="
    
    # Run training in parallel
    parallel -j $POPULATION "
        run_dir=\"gen${gen}_run{1}\";
        echo \"-> Starting \$run_dir\";
        mkdir -p \"\$run_dir\";
        (
            cd \"\$run_dir\";
            if [ -n \"$best_model\" ]; then
                export INIT_MODEL=\"../$best_model\";
            else
                unset INIT_MODEL;
            fi;
            python \"../$TRAIN_SCRIPT\";
        )
    " ::: $(seq 1 $POPULATION)
    
    # Collect candidates
    CANDIDATES=()
    for i in $(seq 1 $POPULATION); do
        run_dir="gen${gen}_run${i}"
        CANDIDATES+=("$run_dir/$MODEL_NAME")
    done
    
    # Evaluate and select best model
    echo "Evaluating models from generation $gen..."
    best_reward=-99999
    for model in "${CANDIDATES[@]}"; do
        if [ -f "$model" ]; then
            reward=$(python "$REFINEMENT_SCRIPT" --model "$model" --eval-only --episodes $EVAL_EPISODES | grep "Average reward:" | awk '{print $3}')
            echo "Model $model: reward $reward"
            if (( $(echo "$reward > $best_reward" | bc -l) )); then
                best_reward=$reward
                best_model=$model
            fi
        fi
    done
    
    echo "Best model from generation $gen: $best_model (reward: $best_reward)"
done
