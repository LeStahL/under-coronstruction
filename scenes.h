#ifndef SCENES_HEADER
#define SCENES_HEADER

#define t_logo210 (0)
#define t_revision (12)
#define t_scaleballs (34.285701751708984)
#define t_cubesausage (54)
#define t_credits (72)
#define duration (99)

const double start_times[] = {
    t_logo210,
    t_revision,
    t_scaleballs,
    t_cubesausage,
    t_credits,
};

const char *scene_names[] = {
    "Logo 210",
    "Revision Graffiti",
    "Scale Balls",
    "Cube Sausage",
    "Credits",
};

const unsigned int nscenes = ARRAYSIZE(start_times);

// We need these two arrays to always have the same size - the following line will cause a compiler error if this is ever not the case
_STATIC_ASSERT(ARRAYSIZE(start_times) == ARRAYSIZE(scene_names));

#endif
