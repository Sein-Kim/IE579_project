## Proposed methods

We modify the agent_rl.py in python/agent/agent_rl folder.

(We does not modify any code from instruction code of lecture except agent_rl.py)

Therefore, just replacing the agent_rl.py code, in your environment, you can conduct our proposed model.

## Train models
To train our model, use

```
python python/battle_train.py --learning_rate 0.001
```

## Eval models
To evaluate our model-agent1 againts to Random model-agent2, use

```
python python/battle_eval.py
```