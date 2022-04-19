# CONCERT
Context-of-Sequences model for predicting DNA Replication Timing signals and identifying sequence elemetns that modulate DNA replication timing

CONCERT is an interpretable context-attentive model to simultaneously identify predictive genomic loci that are potentially important for modulating DNA replication timing (RT) and predict the genome-wide RT profiles from DNA sequences features.

The CONCERT model is structured with two functionally cooperative modules: (1) Selector and (2) Predictor, which are trained jointly to model long-range spatial dependencies across genomic loci, detect predictive genomic loci, and learn context-aware feature presentations of genomic sequences. The selector is targeted at estimating which genomic loci are of potential importance for modulating the RT profiles, approximately selecting a set of predictive genomic loci by importance estimation-based sampling. Leveraging the sequence importance estimation from the selector, 
the predictor performs selective learning of spatial dependencies across genomic loci, to make prediction of RT signals over large-scale spatial domains.
