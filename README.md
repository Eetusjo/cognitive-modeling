# Cognitive Modeling
### Experiments in embodying Pong agents.

Implements and experiments with several methods of "embodying" and making a RL pong agent more cognitively and biologically plausible. Some of the following are WIP.

* Movement constraints [x], energy levels [ ]
* Restricted field of vision, simulating foveal/peripheral vision [x]
* Hierarchical visual processing [testing]
* Delays in perception and action [ ]
* Inner states, predictive coding [ ]

### Usage

The major dependencies are [Keras](https://keras.io) for the policy network, [Gym](https://gym.openai.com) for the Pong environment, [NumPy](http://www.numpy.org) for handling rewards/general computations, [OpenCV](https://opencv.org) for the visual restrictions, and [tensorboardX](https://tensorboardx.readthedocs.io/en/latest/index.html) for logging and visualizing training. All of the dependencies can be installed with `pip install -r requirements.txt` (or equivalent for your environment manager of choice).

You train a model simply by running the script `pong-embodied.py`:

```python pong-embodied.py -n {experiment-name}```

The script admits several arguments for experimenting with different training parameters and restrictions. As usual, run `python pong-embodied.py -h` for details.
