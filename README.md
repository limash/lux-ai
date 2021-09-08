# Lux-ai

### Stage 1 - imitate Ilia rule based agent

#### Initial attempt

The model consumes unified observations and predicts actions for all entities (units and city tiles) one by one.
Example: We have 2 workers and 1 city tile. 
The model consumes an observation constructed for the first unit and predicts its action.
Then it consumes an observation for the second unit and predicts its action.
The model consumes an observation for the city tile and predicts its action.
Predictions can be done in a batch.

**Observation**: 
An unified array of size (32, 32, 57).
The width and length are 32 and 32; they are permanent.
If an actual map is smaller, pool with zeros to 32x32.
(:, :, 0) - first feature layer, (:, :, 1) - second feature layer, etc.
0 to 3 feature layers: the header (first 4 feature layers), it defines whether an observation refers to worker, cart, 
or city tile.
4 to 16 feature layers: units info;
17 to 23 feature layers: city tiles info:
24 to 36 feature layers: common info;
37 to 56: map info.
The model according to the header should recognize a type of entity.

**Actions**: 
A unified one-hot vector of size 40 for workers, carts, city tiles. 
0 to 3 positions are for city tiles (actions: build worker, build cart, research, idle).
4 to 22 positions  are for workers.
23 to 39 are for carts.

**The model**: 
Two input parameters: an observations array and an action mask vector.
The stem - the model from HandyRL agent from Geese competition.
Observation are processed by the stem.
The stem has in the end 3 MLP layers with softmax activations, for each type of entity.
The sizes of MLP layers similar to the action vectors: 4, 19, 17 for city tiles, workers, carts correspondingly.
These layers predict probabilities of actions.
Then they concatenated and multiplied by the action mask vector. 
The action mask vector corresponds to the entity type of the input observation.
For example: if an actual observation is for a city tile, the model should predict an action for a city tile.
Action mask vector will be [1, 1, 1, 1, 0, 0, ..., 0] - 40 positions in total.
Multiplying by this vector filters probabilities of other entity types.

**The replay storage**:
The replay buffer is actually several tfrecord files.
Episodes are stored in the replay buffer.
Episodes consist of n_points.
One n_point contains (action, action_probs, action_mask, observation, reward, temporal_mask, progress).
Action is a response for the current observation; reward, done are for the current observation.
Reward is +1 if the current trajectory is winning.
It is -1 if it is losing.
0 reward for the draw.
Progress is a vector from 0 to 1.
`progress = tf.linspace(0., 1., final_episode_step + 2)[:-1]`

**Result**:
Maximum categorical accuracy is around 0.7.
Ilia policy has some decisions should be done immediately after the entity appearance.
The model cannot catch it since the current observation array does not have such information.
The model is not idle, but it does not reproduce the general behaviour.

**Notes**:
The current lines after the last convolutional layer are quite important.
```
z = layers.Multiply()([z, x[:, :, :, :1]])
z = layers.MaxPooling2D(32)(z) 
z = layers.Flatten()(z)
```
They multiply by zeros all values in the output layer except the position of the entity.
Here z and x are (batch size, 32, 32, 64).
After flattening z is a dense vector of size 64.
Replacing it with downsampling with convolutions with strides, or something like 
`z = layers.Conv2D(filters=64, kernel_size=32)(z)` drops the productivity twice.
