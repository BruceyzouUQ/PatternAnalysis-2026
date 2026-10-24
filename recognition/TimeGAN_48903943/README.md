# TimeGAN - a Limit Order Book Simulation

## Introduction
a description of the algorithm and the problem that it solves

## Feasibility Review

**1. User Need, Scope & Acceptance Criteria**

Synthetic LOB (limit order book) data is valuable for quantitative stock traders who are 
interested in developing and validating effective algorithmic strategies. It allows traders to 
evaluate their models on a plausible imitation before taking actual financial risks. Training on 
only real-world data can lead to overfitting on one historical dataset and training on synthetic 
data can help mitigate this by introducing variation on the observed market.

Plausible LOB data must follow a set of acceptance criteria: no crossed or locked books exist 
(the best bid must always be lower than the best ask), the ask ladder must monotonically 
increase with level, the bid ladder must monotonically decrease with level, volumes of all ask 
and bid sizes must be non-negative, produce authentic volatility clustering (big moves are 
followed by big moves and small moves are followed by small moves), produce heavy-tailed 
return distributions (clear positive excess kurtosis and tail quantiles match the real data to 
some degree), have a KL (Kullback-Leibler) divergence of ≤ 0.1 on spread and midprice returns 
and a SSIM (structural similarity) > 0.6 on depth heatmaps (price levels × time steps).

**2. Model Choice & Course Concepts**

The model of choice will be a generative adversarial network (GAN), specifically a TimeGAN. A 
TimeGAN extends the traditional GAN by introducing the embedding/recovery space and two 
important losses: LS (supervised loss), and LR (ensures the embedding preserves enough 
information needed to accurately reconstruct real data, allowing latent representations to be 
meaningfully decoded) (Yoon et al., 2019). The goal of the TimeGAN is to preserve temporal 
dynamics by learning from both real history and the entire generated sequence (using LU) to pick 
up on transitional and holistic failure.

As for each module’s architecture, the embedding network, generator and discriminator will 
likely need to be RNNs (recurrent neural networks) whereas the recovery network can be 
feedforward because each latent vector already has the temporal context needed for 
reconstruction (Yoon et al., 2019).

The LOBSTER AMZN Level 10 dataset itself features 10 price levels where level one is the lowest 
ask and the highest bid; level two is the second lowest ask and the second highest bid and so on.
Each level contains ask price, ask volume, bid price and bid volume, totalling 40 columns 
(number of details per level × number of levels) for every entry. Each entry marks an event that 
occurred at a specific time.

The TimeGAN is a hard difficulty project.

**3. Preliminary Feasibility Evidence**

As a baseline, a TimeVAE inspired model has been created. Running this model on the sample data 
(AMZN level 10 data on one day) and over five epochs as a smoke test, the model successfully 
exhibited the ability to learn. Its validation loss began at 1831.0 and dropped to 1612.0 in 
these five epochs which took 4 mins and 2 secs. This smoke test reveals that the dataset is 
workable.

**4. Risks, Budget & Fallback**

The main risk of a TimeGAN is its adversarial nature possibly leading to unstable training and 
collapse. In GANs, the generator can possibly overpower the discriminator causing the 
discriminator to recognise every sample as real, or the discriminator can overpower the 
generator where the discriminator successfully catches every fake sample leaving no opportunity 
for the generator to learn what’s realistic. Due to this difficultly, training time may be long 
and could ultimately result in failure.

Computing budget wise, training should be able to succeed on one rangpur GPU with 16GB of 
memory. Runtimes will be limited in SLURM scripts and will be adjusted appropriately (with 
buffer) based on epochs. 

The next planned experiment will be running the baseline model for more epochs and visualising 
its samples. If these are successful according to the acceptance criteria discussed in section 1., 
development of the TimeGAN can begin.

A realistic fallback in case of the TimeGAN’s failure or unfeasible training requirements, would 
be to simplify the architecture proposed by Yoon et al. and instead return to just a 
generator-discriminator model where time series relationships are still learnt. This would be 
more similar to the C-RNN-GAN or RCGAN that Yoon et al. proposed.