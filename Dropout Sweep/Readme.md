## Description

The effect of dropout when swept was evaluated by running the original misra experiment for the gpt-2 model with varying levels of dropout from 0.1 to 0.8. 

In MC dropout literature the ideal dropout is in the neighborhood of 0.1. The question is, is this the case when building psychological populations? To evaluate this, we look to see if the variance for each stimuli in each category grows in a behaved manner with increasing dropout and if the mean remains relatively stable. If this is the case, then it is reasonable to conclude that a dropout of 0.1 is sufficient to characterize the behavior of dropout population of that model species in general. 

The mean is not stable and neighter is the std. Rather the mean probability continually drops and the std increases at an accelerating rate. 

Neither category nor typicality seemed to significantly effect the change or rate of change of mean probability or mean std as dropout increased.
