Hey! build me a framework named Primal that unifies:
1. Active Inference (no other thing excepts "Change the world to match yours, or change yours to match the world", FE, Bayesian Inference, and predictive coding/action generation, because other like full VI planning is expensive and slow, and we just take the idea of Active Inference, which is those are just the main ideas to AIF but idk why friston starts expanding it to make it worse and expensive)
2. Log Space fusion, and other Exponential Conjugate based one for cheap bayesian updating and also for continual learning (growing components, not setting the parameters, so the parameters is minimal on start then expands, just like Bayesian Model Expansion and will not forget until BMR decides to prune/merge it) and rapid adaptation
3. Core Knowledge and Transfer learning for super duper fast learning(Spelke's objects, space andgeometry, number, agents, and physics, and you can add any other you would like to, but the main is Spelke's one)
4. Theory Theory (just take the main idea of multiple beliefs/hypotheses based on current context and prediction and selection instead of using full VI)
5. BMR for merge/prune components
6. PFC and VLPFC (can change temperatures based on prediction errors/FE), Retinas, visual cortex, anterial temporal lobe, Hippocampus/Hippocampal, Hemisphere and bilateral Hemifield split, and other brain mechanism or components you would like to add to make it really good learners (note: don't do the full computation of what brain does, instead we just take the output/simplest how it works function, because we just want the engineering benefit without the actual work brain does while still keeping full benefit)
7. cerebellar smoothing for motor output
8. Weber-Fechner logarithmic ANS precision scaling
9. Superior Colliculus
10. Occipital Lobe
11. hemifield pull imbalance over coordinates
12. precision alpha scaling modeling survival urgency
13. 18-step lattice-boltzmann fluid advection for intuitive physics simulation (for spelke's priors)
14. propriorception for body awarness as a continuous Gaussian
15. Markovian temporal decay on prior belief (0.7 old + 0.3 new)
16. Renormalization Group
17. Common Sense Reasoning
18. and more you would like to add (brainstorm it)

IMPORTANT: must be fully 100% sub symbolic (no hardcoded domain knowledge, no hardcoded policy, no hardcoded prediction, no hints, etc.), it must not know anything/blind about the world (at architecture level), and it must be lightweight (as long as you implement all those as engineering benefits only it will be lightweight) and must be 100% complete implementation of what i asked and you added, before making the code or whatever you must verify that your logic, your math, your whatever works, if not then refine, AND MOST IMPORTANTLY DO NOT OVERTHINK OR OVER SIMPLIFY OR OVERENGINEERS!

btw implement it on python numpy, scipy and whatever lib that is not huge, also add pyproject.toml, our custom LICENSE (our company name is Primeval), you must verify the code works too (not just work on execution test, but logic/code/classes/whatever must works as expected), test it on gymnasium ALE heavy physics game like breakout, pong, cart pole, or whatever (with correct implementation i believe it can master those under 2 episodes or just 3 lives left, track that output, if it doesn't match our expectation for under 2 episode learning, reason/deepthink/brainstorm about what's wrong and debug and fix/refine it), and for image test it on MNIST (maybe 10 samples each number?)

TASK YOU GOTTA DO:
1. implement all 18 in brain/ folder (and the main agent.py file that calls all those files) and also an init
2. write custom Primeval LICENSE
3. write pyproject.toml
4. verify all works on physics and image
5. document README.md
