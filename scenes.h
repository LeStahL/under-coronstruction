#ifndef SCENES_HEADER
#define SCENES_HEADER

#define t_scaleballs (0)
#define t_revision (10)
#define t_cubesausage (20)
#define duration (130)

const double start_times[] = {
    t_scaleballs,
    t_revision,
    t_cubesausage,
};

const char *scene_names[] = {
    "Scale Balls",
    "Revision Graffiti",
    "Cube Sausage",
};

const unsigned int nscenes = ARRAYSIZE(start_times);

// We need these two arrays to always have the same size - the following line will cause a compiler error if this is ever not the case
_STATIC_ASSERT(ARRAYSIZE(start_times) == ARRAYSIZE(scene_names));

#endif
