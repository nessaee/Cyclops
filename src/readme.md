# Cyclops Protocol

## Description

## Installation


```bash
# Clone the repository

# Navigate to the directory

# Install dependencies
pip install -r requirements.txt

## Command Line Arguments
--theta: Activates theta-related functionality.
--kernel: Turns on kernel-related operations.
--x_px: Enables operations related to x_px.
--claimed: Activates functionality related to 'claimed'.
--initialize: Enables the initialization process.
--visible: Sweeps the visible distance value.
--encode: Applies encoding before evaluation.
--threshold: Enables a sweep over alpha.
--scoring: Enables a sweep over beta.
--rate: Enables a sweep over different sample rates.
--sequence: Show sequences during testing
-a or --alpha: Sets the similarity threshold adjustment starting from 0.5 (in hundredths).
-b or --beta: Sets the beta value.
-t or --tau: Sets the tau value.
-K: Sets the number of tests to be conducted.
-k: Specifies the kernel size.
-q or --Q: Defines the maximum state value.
```

## Usage 
python main.py --dist --theta --kernel --x_px --claimed --initialize --visible --encode --threshold --scoring --rate --sequence -a -b -t -K -k -q -s -ab -f -r --save

---

## Understanding the Cyclops Protocol

The Cyclops protocol, aimed at enhancing the security of AV platoons, is a complex system involving various components. This repository plays a crucial role in testing and validating this protocol. Below is an explanation of key components and their contribution to the experimentation involved.

### 1. `main.py` - The Central Controller

- **Purpose**: Acts as the primary driver for the Cyclops protocol.
- **Functionality**: Initializes and orchestrates the various components of testing, managing the flow of operations and interactions between the prover and verifier vehicles.


### 2. `analysis.py` - Data Analysis and Processing

- **Purpose**: Processes and analyzes data collected during the protocol's execution.
- **Functionality**: Involves image processing, feature extraction from camera feeds, and scene comparison for trajectory verification.


### 3. `Adversary.py` - Simulating Security Threats

- **Purpose**: Tests the protocol's resilience against different types of adversaries.
- **Functionality**: Implements remote adversary attack strategies to evaluate the protocol's security measures.


### 4. `Markov.py` - Traffic Pattern Modeling

- **Purpose**: Models traffic environments as a Markov chain, essential for remote adversary prediction of state changes .
- **Functionality**: Generates probabilistic models of traffic flows, aiding in scene-matching and adversary prediction.


### 5. `Matrix.py` - Handling Mathematical Operations

- **Purpose**: Manages mathematical computations involved in image transformations.
- **Functionality**: Used for calculating the cFoV, applying perspective transformations, and aiding in the fuzzy scene-matching algorithm.


### 6. `Protocol.py` - Protocol Logic Implementation

- **Purpose**: Encapsulates the Cyclops protocol's logic.
- **Functionality**: Defines the step-by-step process of the protocol, from digital identity verification to physical identity verification.


### 7. `Region.py` - Field of View Partitioning

- **Purpose**: Manages the division of the field of view into regions for scene matching.
- **Functionality**: Responsible for partitioning images and aiding in scene analysis.


### 8. `Sequence.py` - Image and Data Sequencing

- **Purpose**: Ensures the proper sequencing of collected images and data.
- **Functionality**: Manages timestamping, ordering, and processing of scenes for trajectory verification.


---

### Importance 

Each script is tailored to handle specific aspects of the Cyclops protocol, from initiating the process to analyzing data and simulating adversarial attacks. Together, they form a comprehensive system for testing and validating the protocol's effectiveness. This modular approach also allows for the isolation and refinement of individual components, making the protocol more robust and adaptable to different scenarios and threats.

By employing these scripts in simulations and real-world testing environments, the Cyclops protocol can be rigorously evaluated, ensuring its reliability and effectiveness in protecting the digital and physical identities of autonomous vehicles.