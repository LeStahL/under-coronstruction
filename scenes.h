#ifndef SCENES_HEADER
#define SCENES_HEADER

#define t_revision (0)
#define duration (180)

const double start_times[] = {
    t_revision,
};

const char *scene_names[] = {
    "Revision Graffiti",
};

const unsigned int nscenes = ARRAYSIZE(start_times);

// We need these two arrays to always have the same size - the following line will cause a compiler error if this is ever not the case
_STATIC_ASSERT(ARRAYSIZE(start_times) == ARRAYSIZE(scene_names));

#endif
