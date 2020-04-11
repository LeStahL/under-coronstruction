#ifndef SCENES_HEADER
#define SCENES_HEADER

#define t_logo210 (0)
#define t_revision (12)
#define t_scaleballs (35.83209991455078)
#define t_cubesausage (58.68920135498047)
#define t_greets (81.54630279541016)
#define t_credits (92.97489929199219)
#define duration (101)

const double start_times[] = {
    t_logo210,
    t_revision,
    t_scaleballs,
    t_cubesausage,
    t_greets,
    t_credits,
};

const char *scene_names[] = {
    "Logo 210",
    "Revision Graffiti",
    "Scale Balls",
    "Cube Sausage",
    "Greetings",
    "Credits",
};

const unsigned int nscenes = ARRAYSIZE(start_times);

// We need these two arrays to always have the same size - the following line will cause a compiler error if this is ever not the case
_STATIC_ASSERT(ARRAYSIZE(start_times) == ARRAYSIZE(scene_names));

#endif
