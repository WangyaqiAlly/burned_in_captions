PURPOSE
=======

Validates that GAN learns the data rather than specific tasks.

DATA
====

Celeb-A, all 202599 files are split into:
         - 10% (20236) in "test" directory
         - 90% (182363) training data
               - all (182363) in "train-unlabelled" directory
               - 10% (18171) in "train-labelled" directory

GAN
===

We learn the data once using GAN (CT-GAN).  In different experiments,
the GAN is trained sing the following conditioning:
   - Smiling
   - High-Cheekbone
   - Male
   - Unconditioned

Tasks
=====

We focus on "Smiling" and "High-Cheekbone".

We want to test classifier results for the two tasks for the four GAN above.

Commands:
=========

Below examples 