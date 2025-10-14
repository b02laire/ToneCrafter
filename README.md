<h1 align=left><img src="ToneCrafter_logo.png" width="60">&emsp;ToneCrafter</h1>

ToneCrafter started as a project developed by students at [ENSEA](https://www.ensea.fr/). You can
still see our work on the official repo or on the legacy branch of this one.  

Since the project ended, AI/ML applied to audio have evolved, and projects such
as [RAVE](https://github.com/acids-ircam/RAVE) could be used to produce results
we could not even dream of a few years ago.  
My goal here is no to develop the perfect solution, since "just" training an 
instance of RAVE would not be that interesting, but rather work my way up by 
following "naïve" approaches we did not pursue back then.

The current plan is to develop four networks:  

- a MLP, to test stuff like data ingestion, loss functions etc
- a CNN, with a different data ingestion system from our OG model.
- a VAE applied to actual music.
- a GAN, though more demanding in terms of data and computation power.

Depending on time and resources, a hybrid approach (VAE+GAN) like the one 
used by RAVE could be tested, I am also interested in using DDSP.

## FAQ:

1. What dataset do you plan on using ?  
   As of now, IDMT-SMT-Audio-Effects seems like the way to go. I might have to 
   make my own depending on how it performs.

2. How will your networks ingest audio ?  
    I want to explore something different from MFCC/STFT only. Perhaps a Pseudo
    QMF decomposition (like RAVE, again) would bear better results.


## Resources:
- https://andrew.gibiansky.com/pqmf-subband/

    