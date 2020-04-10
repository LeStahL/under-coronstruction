#ifndef DRAW_HEADER
#define DRAW_HEADER

if(t < t_revision)
{
    glUseProgram(shader_program_gfx_logo210.handle);
    glUniform1f(shader_uniform_gfx_logo210_iTime, t-t_logo210);
    glUniform2f(shader_uniform_gfx_logo210_iResolution, w, h);
#ifdef MIDI
    glUniform1f(shader_uniform_gfx_logo210_iFader0, fader0);
    glUniform1f(shader_uniform_gfx_logo210_iFader1, fader1);
    glUniform1f(shader_uniform_gfx_logo210_iFader2, fader2);
    glUniform1f(shader_uniform_gfx_logo210_iFader3, fader3);
    glUniform1f(shader_uniform_gfx_logo210_iFader4, fader4);
    glUniform1f(shader_uniform_gfx_logo210_iFader5, fader5);
    glUniform1f(shader_uniform_gfx_logo210_iFader6, fader6);
    glUniform1f(shader_uniform_gfx_logo210_iFader7, fader7);
#endif
}
else if(t < t_scaleballs)
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
else if(t < t_cubesausage)
{
    glUseProgram(shader_program_gfx_scaleballs.handle);
    glUniform1f(shader_uniform_gfx_scaleballs_iTime, t-t_scaleballs);
    glUniform2f(shader_uniform_gfx_scaleballs_iResolution, w, h);
#ifdef MIDI
    glUniform1f(shader_uniform_gfx_scaleballs_iFader0, fader0);
    glUniform1f(shader_uniform_gfx_scaleballs_iFader1, fader1);
    glUniform1f(shader_uniform_gfx_scaleballs_iFader2, fader2);
    glUniform1f(shader_uniform_gfx_scaleballs_iFader3, fader3);
    glUniform1f(shader_uniform_gfx_scaleballs_iFader4, fader4);
    glUniform1f(shader_uniform_gfx_scaleballs_iFader5, fader5);
    glUniform1f(shader_uniform_gfx_scaleballs_iFader6, fader6);
    glUniform1f(shader_uniform_gfx_scaleballs_iFader7, fader7);
#endif
}
else if(t < t_credits)
{
    glUseProgram(shader_program_gfx_cubesausage.handle);
    glUniform1f(shader_uniform_gfx_cubesausage_iTime, t-t_cubesausage);
    glUniform2f(shader_uniform_gfx_cubesausage_iResolution, w, h);
#ifdef MIDI
    glUniform1f(shader_uniform_gfx_cubesausage_iFader0, fader0);
    glUniform1f(shader_uniform_gfx_cubesausage_iFader1, fader1);
    glUniform1f(shader_uniform_gfx_cubesausage_iFader2, fader2);
    glUniform1f(shader_uniform_gfx_cubesausage_iFader3, fader3);
    glUniform1f(shader_uniform_gfx_cubesausage_iFader4, fader4);
    glUniform1f(shader_uniform_gfx_cubesausage_iFader5, fader5);
    glUniform1f(shader_uniform_gfx_cubesausage_iFader6, fader6);
    glUniform1f(shader_uniform_gfx_cubesausage_iFader7, fader7);
#endif
}
else {
    glUseProgram(shader_program_gfx_credits.handle);
    glUniform1f(shader_uniform_gfx_credits_iTime, t-t_credits);
    glUniform2f(shader_uniform_gfx_credits_iResolution, w, h);
#ifdef MIDI
    glUniform1f(shader_uniform_gfx_credits_iFader0, fader0);
    glUniform1f(shader_uniform_gfx_credits_iFader1, fader1);
    glUniform1f(shader_uniform_gfx_credits_iFader2, fader2);
    glUniform1f(shader_uniform_gfx_credits_iFader3, fader3);
    glUniform1f(shader_uniform_gfx_credits_iFader4, fader4);
    glUniform1f(shader_uniform_gfx_credits_iFader5, fader5);
    glUniform1f(shader_uniform_gfx_credits_iFader6, fader6);
    glUniform1f(shader_uniform_gfx_credits_iFader7, fader7);
#endif
}
#endif
