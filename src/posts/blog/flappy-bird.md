---
title: "Flappy Bird Using Reinforcement Learning"
category: "Machine Learning"
date: "2024-03-13T23:46:37.121Z"
desc: "Flappy bird training using RL"
thumbnail: "./images/code-block/thumbnail.jpg"
alt: "code block graphic"
---

## Introduction to Reinforcement Learning

Reinforcement Learning (RL) is an area of machine learning concerned with how agents ought to take actions in an environment to maximize some notion of cumulative reward. In RL, an agent interacts with the environment in a loop:

1. **Observation**: The agent observes the current state of the environment.
2. **Action**: Based on the observation, the agent chooses an action.
3. **Reward**: The environment transitions to a new state, and the agent receives a reward associated with the transition.

The goal of the agent is to learn a policy ![Policy](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\pi) that maximizes the expected cumulative reward, often called the return ![Return](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;G_t=R_{t+1}+\gamma&space;R_{t+2}+\gamma^2&space;R_{t+3}+\ldots).

Here, ![Reward](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;R_{t+1}) is the reward received after taking action ![Action](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;a_t) in state ![State](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;s_t), and ![Gamma](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\gamma) (0 ≤ ![Gamma](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\gamma) < 1) is the discount factor, which models the agent's consideration of future rewards.

## Key Concepts in Reinforcement Learning

- **State (s)**: A representation of the current situation in the environment.
- **Action (a)**: A decision or move made by the agent.
- **Policy (![Policy](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\pi))**: A mapping from states to probabilities of selecting each possible action.
- **Reward (r)**: Feedback from the environment used to evaluate the action taken.
- **Value Function (![Value Function](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;V(s)>))**: The expected return starting from state ![State](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;s) and following the policy ![Policy](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\pi).
- **Q-Function (![Q-Function](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;Q(s,%20a)>))**: The expected return starting from state ![State](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;s), taking action ![Action](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;a), and thereafter following policy ![Policy](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\pi).

## Types of RL Algorithms

### Q-Learning

Q-Learning is a model-free, off-policy RL algorithm. It estimates the value of taking an action ![Action](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;a) in a state ![State](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;s) and follows the Bellman equation for updating the Q-values:

![Q-Learning Equation](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;Q(s,%20a)%20\leftarrow%20Q(s,%20a)%20+%20\alpha%20\left[%20r%20+%20\gamma%20\max_{a'}%20Q(s',%20a')%20-%20Q(s,%20a)%20\right]>)

Here, ![Alpha](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\alpha) is the learning rate, ![Reward](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;r) is the immediate reward, ![Next State](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;s') is the next state, and ![Gamma](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\gamma) is the discount factor.

### Deep Q-Networks (DQN)

DQN extends Q-learning by using a deep neural network to approximate the Q-function. The neural network takes the state as input and outputs Q-values for all possible actions.

Key innovations in DQN include:

- **Experience Replay**: Storing past experiences and sampling them randomly during training to break correlations between consecutive experiences.
- **Target Network**: A separate network for estimating the target Q-value, which is periodically updated with the weights of the main network to stabilize training.

### Policy Gradient Methods

Policy gradient methods directly optimize the policy by computing the gradient of the expected reward concerning the policy parameters ![Theta](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\theta):

![Policy Gradient Equation](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\nabla_{\theta}%20J(\theta)%20=%20\mathbb{E}_{\pi_{\theta}}%20\left[%20\nabla_{\theta}%20\log%20\pi_{\theta}(a%20|%20s)%20Q^{\pi_{\theta}}(s,%20a)%20\right]>)

The policy ![Policy Theta](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\pi_{\theta}) is often parameterized by a neural network, and the gradient ascent method is used to improve the policy.

### Proximal Policy Optimization (PPO)

PPO is an advanced policy gradient method that introduces a clipped objective to prevent the policy from updating too far in a single step, balancing exploration and exploitation. The PPO objective function is given by:

![PPO Objective](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;L^{CLIP}(\theta)%20=%20\mathbb{E}_t%20\left[%20\min%20\left(%20r_t(\theta)%20\hat{A}_t,%20\text{clip}(r_t(\theta),%201%20-%20\epsilon,%201%20+%20\epsilon)%20\hat{A}_t%20\right)%20\right]>)

Where ![r_t(theta)](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;r_t(\theta)>) is the ratio of the new policy to the old policy, ![Advantage](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\hat{A}_t) is the advantage function, and ![Epsilon](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\epsilon) is a small hyperparameter that controls the clipping range.

## The Flappy Bird Game

![Flappy Bird Gameplay](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnFiODkwbzN1aDFmdG85aGk0a3RlMTFidGpmN2t4azNoODJqM2M0dyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/kKj5Olw5JobOhiFVHs/giphy.gif)

Flappy Bird is a side-scrolling game where the player controls a bird attempting to fly between sets of pipes without colliding with them. The game is deceptively simple but highly challenging, making it an ideal candidate for applying RL.

## Game Details

- **State Space**: The state of the game can be represented by features such as the bird's vertical position, its velocity, the distance to the next pipe, and the position of the gap in the next pipe.
- **Action Space**: The action space is discrete, with two possible actions: flap (which makes the bird ascend) or do nothing (which causes the bird to descend).
- **Reward System**: Designing an appropriate reward system is crucial. Possible rewards include:
  - A positive reward for successfully passing through a gap.
  - A negative reward for colliding with a pipe or the ground.
  - A small negative reward for each timestep to encourage the bird to move forward rather than hover in place.

## Mathematical Formulation

In the context of Flappy Bird, we define the reward function ![Reward Function](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;r(s,%20a)>) based on the outcomes of each action:

- If the bird passes through a pipe, ![Positive Reward](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;r(s,%20a)%20=%20+1>).
- If the bird hits a pipe or the ground, ![Negative Reward](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;r(s,%20a)%20=%20-1>).
- For each frame survived, a small negative reward can be given to encourage the bird to make progress, for example, ![Timestep Reward](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;r(s,%20a)%20=%20-0.01>).

The Q-values can be updated using the Q-learning rule:

![Q-Learning Update](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;Q(s_t,%20a_t)%20\leftarrow%20Q(s_t,%20a_t)%20+%20\alpha%20\left[%20r_{t+1}%20+%20\gamma%20\max_{a'}%20Q(s_{t+1},%20a')%20-%20Q(s_t,%20a_t)%20\right]>)

## Implementation Details

To implement the Flappy Bird environment and train an RL agent, we'll leverage OpenAI Gym, a toolkit that provides the standard API for interfacing with various environments.

### Custom Environment

The custom Flappy Bird environment is implemented in the `flappyenv.py` file. It inherits from the `gym.Env` class and implements the necessary methods (`reset()`, `step()`, etc.) to interact with the agent.

#### Key Components

Below are the key components of the Flappy Bird game implemented using Pygame:

1. **Bird Class**: Represents the bird controlled by the agent.

   ```python
   # Bird class
   class Bird:
       def __init__(self):
           self.x = 50
           self.y = SCREEN_HEIGHT // 2
           self.velocity = 0

       def draw(self):
           screen.blit(bird_image, (self.x, self.y))

       def update(self):
           self.velocity += GRAVITY
           self.y += self.velocity

       def jump(self):
           self.velocity = JUMP_STRENGTH

       def fall_faster(self):
           self.velocity += FALL_STRENGTH

       def get_rect(self):
           return pygame.Rect(self.x, self.y, BIRD_WIDTH, BIRD_HEIGHT)
   ```

   **Explanation:**

   - **Initialization**: Sets the bird's starting position and initial velocity.
   - **Draw Method**: Renders the bird image on the screen.
   - **Update Method**: Applies gravity to the bird, updating its vertical position.
   - **Jump Method**: Makes the bird ascend by setting a negative velocity.
   - **Fall Faster Method**: Increases the bird's velocity to make it descend more quickly.
   - **Get Rect Method**: Returns the bird's rectangle for collision detection.

2. **Pipe Class**: Represents the obstacles the bird must navigate through.

   ```python
   # Pipe class
   class Pipe:
       def __init__(self):
           self.x = SCREEN_WIDTH
           self.height = random.randint(100, 400)
           self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
           self.bottom_rect = pygame.Rect(self.x, self.height + PIPE_GAP, PIPE_WIDTH, SCREEN_HEIGHT)

       def draw(self):
           pygame.draw.rect(screen, GREEN, self.top_rect)
           pygame.draw.rect(screen, GREEN, self.bottom_rect)

       def update(self):
           self.x -= PIPE_SPEED
           self.top_rect.x = self.x
           self.bottom_rect.x = self.x
   ```

   **Explanation:**

   - **Initialization**: Sets the pipe's starting position and randomly determines the height of the top pipe.
   - **Draw Method**: Renders the top and bottom pipes on the screen.
   - **Update Method**: Moves the pipes to the left, simulating the bird's forward movement.

3. **Main Game Loop**: Handles game events, updates game objects, and renders the game.

   ```python
   # Main game loop
   def main():
       clock = pygame.time.Clock()
       bird = Bird()
       pipes = [Pipe()]
       score = 0
       running = True

       while running:
           for event in pygame.event.get():
               if event.type == pygame.QUIT:
                   pygame.quit()
                   sys.exit()
               if event.type == pygame.KEYDOWN:
                   if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                       bird.jump()
                   if event.key == pygame.K_DOWN:
                       bird.fall_faster()
           # Update game objects
           bird.update()
           for pipe in pipes:
               pipe.update()

           # Check for collisions
           bird_rect = bird.get_rect()
           if bird.y < 0 or bird.y > SCREEN_HEIGHT:
               running = False
           for pipe in pipes:
               if pipe.top_rect.colliderect(bird_rect) or pipe.bottom_rect.colliderect(bird_rect):
                   running = False

           # Add new pipes
           if pipes[-1].x < SCREEN_WIDTH - 200:
               pipes.append(Pipe())

           # Remove off-screen pipes
           if pipes[0].x < -PIPE_WIDTH:
               pipes.pop(0)
               score += 1

           # Draw everything
           screen.blit(background_image, (0, 0))  # Draw background
           bird.draw()
           for pipe in pipes:
               pipe.draw()

           # Draw score
           font = pygame.font.Font(None, 36)
           score_text = font.render(f"Score: {score}", True, BLACK)
           screen.blit(score_text, (10, 10))

           pygame.display.flip()
           clock.tick(FPS)

   if __name__ == "__main__":
       main()
   ```

   **Explanation:**

   - **Event Handling**: Listens for user inputs to control the bird's actions.
   - **Updating Objects**: Updates the bird's position and moves the pipes.
   - **Collision Detection**: Checks if the bird has collided with pipes or gone out of bounds.
   - **Pipe Management**: Adds new pipes and removes pipes that have moved off-screen, updating the score accordingly.
   - **Rendering**: Draws the background, bird, pipes, and score on the screen.
   - **Game Loop Control**: Ensures the game runs at the specified frames per second (FPS) and handles quitting the game.

### Agent Training

Training involves running episodes where the agent interacts with the environment, collecting experiences, and updating its policy or Q-values based on the observed rewards. Over time, the agent learns to avoid obstacles and navigate through the pipes more effectively.
Depending on the algorithm, we will define the neural network architecture, loss functions, and training loop.
We used PPO algorithm in this case.

```python
# Directories for saving models and logs
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Initialize and check the environment
env = FlappyBirdEnv()
check_env(env, warn=True)
env = DummyVecEnv([lambda: env])

# Initialize the PPO model
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# Training loop
TIMESTEPS = 1000000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
```

#### Example: Q-Learning Update in Code

Here's how the Q-learning update rule is implemented in the code:

![Q-Learning Update](<https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;Q(s_t,%20a_t)%20\leftarrow%20Q(s_t,%20a_t)%20+%20\alpha%20\left[%20r_{t+1}%20+%20\gamma%20\max_{a'}%20Q(s_{t+1},%20a')%20-%20Q(s_t,%20a_t)%20\right]>)

**Explanation:**

- **Q-Value Update**: Adjusts the Q-value for the current state-action pair based on the received reward and the maximum expected future rewards.
- **Learning Rate (![Alpha](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\alpha))**: Determines how much new information overrides old information.
- **Discount Factor (![Gamma](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\gamma))**: Balances immediate and future rewards.

### Challenges and Solutions

- **Exploration vs. Exploitation**: Balancing exploration (trying new actions) and exploitation (choosing the best-known action) is crucial. Techniques like ![Epsilon-Greedy](https://latex.codecogs.com/png.image?\inline&space;\dpi{110}\bg{white}&space;\epsilon-greedy) in Q-learning or entropy regularization in PPO can help.
- **Stabilizing Training**: In DQN, techniques like experience replay and target networks help stabilize the training process, preventing the network from diverging.

## Conclusion

Reinforcement Learning offers a robust framework for training agents to perform tasks like playing games autonomously. By applying RL to the Flappy Bird game, we explored key concepts such as Q-learning, DQN, and PPO, and discussed the importance of reward system design in guiding the agent's learning process.

With the help of OpenAI Gym, implementing and experimenting with RL algorithms becomes more accessible, allowing us to train an agent to successfully navigate the challenging environment of Flappy Bird. The journey of training an RL agent involves balancing exploration and exploitation, stabilizing training, and fine-tuning rewards.
