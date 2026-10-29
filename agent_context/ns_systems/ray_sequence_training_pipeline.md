### Discussed so far

- There has been considerable progress when it comes to other discovery methods, i.e we are clearly not the first who's exploring discovery mechanisms. for geometies , fiold geometries. 
- it started out with automated freeform design paper in 2017:
    https://www.nature.com/articles/lsa201781

    In summary, this paper took a rough geometric sketch of the ray path and the surface configuration in spherical surface parameterization and automated the freeform refinement process

- Following this, there have been papers that looked at the solution space of three mirror systems exclusively:


### What we've proposed so far
A three part system
1) Ray path proposing mechanism
2) Compaction model
3) refinement model


### Datasets that are available: 


### Dataset generation methodologies:
- GA paper (Su et al.): 
System parameters + folding geometry + freeform systems -> learn a conditional distribution
Use noise to produce multiple designs


GAN Paper:
- Folding topologies are not invented. 
1) predefine folding "classes" (whats that?)
2) sample a geometry
3) Reject obviously obscured layout
4) construct a spheric based starting point
5) convert to freeform
6) optimize every sample

What I'm interested in their dataset generation part, is
a) extent of reliance on existing folding topologies
b) and the rate at which the data generation process encounters "obscured" designs

So they first divide the design variables into:
- Conditional/system params