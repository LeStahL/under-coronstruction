#ifndef DRAW_HEADER
#define DRAW_HEADER

{
    glUseProgram(shader_program_gfx_revision.handle);
    glUniform1f(shader_uniform_gfx_revision_iTime, t-t_revision);
    glUniform2f(shader_uniform_gfx_revision_iResolution, w, h);
#ifdef MIDI
    glUniform1f(shader_uniform_gfx_revision_iFader0, fader0);
    glUniform1f(shader_uniform_gfx_revision_iFader1, fader1);
    glUniform1f(shader_uniform_gfx_revision_iFader2, fader2);
    glUniform1f(shader_uniform_gfx_revision_iFader3, fader3);
    glUniform1f(shader_uniform_gfx_revision_iFader4, fader4);
    glUniform1f(shader_uniform_gfx_revision_iFader5, fader5);
    glUniform1f(shader_uniform_gfx_revision_iFader6, fader6);
    glUniform1f(shader_uniform_gfx_revision_iFader7, fader7);
#endif
}
#endif
