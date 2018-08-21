#include <cstdlib>
#include <cstdint>
#include <cfloat>
#include <ctime>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <iostream>
//
#define GLM_FORCE_EXPLICIT_CTOR
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/detail/type_vec.hpp>
#include <glm/gtc/noise.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtx/euler_angles.hpp>
//
#include <glad/glad.h>
#include <GLFW/glfw3.h>
//
#include "tick_clock.hpp"

using namespace glm;

typedef int8_t   I8;
typedef int16_t  I16;
typedef int32_t  I32;
typedef uint8_t  U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef float    F32;
typedef double   F64;

constexpr U32   win_scale = 5;
constexpr uvec2 win_size = {320, 180};
constexpr U32   win_volume = win_size.x * win_size.y;
constexpr uvec3 map_size = {128, 128, 128};
constexpr U32   map_volume = map_size.x * map_size.y * map_size.z;

constexpr F32 TAU = pi<F32>() * 2.f;

struct Voxel
{
    vec3 col;
    F32  a;
};

constexpr Voxel border_voxel = {{0.7f, 0.7f, 0.8f}, 1.0f};
constexpr Voxel empty_voxel  = {{0.0f, 0.0f, 0.0f}, 0.01f};
constexpr Voxel fog_voxel    = {{0.7f, 0.7f, 0.8f}, 0.01f};
constexpr Voxel water_voxel  = {{0.1f, 0.3f, 0.6f}, 0.05f};
constexpr Voxel sand_voxel   = {{1.0f, 1.0f, 0.6f}, 1.0f};
constexpr Voxel stone_voxel  = {{0.4f, 0.4f, 0.4f}, 1.0f};
constexpr Voxel snow_voxel   = {{0.9f, 0.9f, 1.0f}, 1.0f};
constexpr Voxel grass_voxel  = {{0.3f, 0.8f, 0.2f}, 1.0f};
constexpr Voxel dirt_voxel   = {{0.6f, 0.3f, 0.1f}, 1.0f};
constexpr Voxel player_voxel = {{1.0f, 0.0f, 1.0f}, 1.0f};

Voxel       map[map_volume];
tvec3<U8>   cast_plane[win_volume];
GLFWwindow *win;

F32   player_yaw = TAU / 3.f;
F32   player_pitch = 0;
ivec3 player_cursor = {-1, -1, -1};
vec3  player_vel = {0, 0, 0};
vec3  player_pos = {map_size.x / 2, map_size.y / 2, 70};

template<typename T>
bool outside_bounds(tvec3<T> const &pos)
{
    return pos.x < 0 || pos.y < 0 || pos.z < 0 ||
           pos.x >= (I32)map_size.x ||
           pos.y >= (I32)map_size.y ||
           pos.z >= (I32)map_size.z;
}

U32 get_map_idx(uvec3 const &pos)
{
    return pos.x | (pos.y << 7) | (pos.z << 14);
}

Voxel const &get_voxel(ivec3 const &pos)
{
    if(outside_bounds(pos)) return border_voxel;
    return map[get_map_idx((uvec3)pos)];
}

void init_gl()
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    win = glfwCreateWindow(win_size.x * win_scale,
                           win_size.y * win_scale, "raycaster", NULL, NULL);
    glfwMakeContextCurrent(win);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(win, GLFW_STICKY_MOUSE_BUTTONS, 1);
    glfwSwapInterval(0);

    if(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0)
    {
        throw std::runtime_error("GLAD error");
    }

    glViewport(0, 0, win_size.x * win_scale, win_size.y * win_scale);
}

F32 randf()
{
    return (F32)(std::rand() % 0x10000) / 0x10000;
}

F32 randf(F32 dist)
{
    return (randf() / dist) + ((1.f - (1.f / dist)) / 2.f);
}

F32 randf(F32 min, F32 max)
{
    return (randf() * (max - min)) + min;
}

F32 blur(F32 val, F32 dist)
{
    return std::clamp(val += (randf() / dist) - (0.5f / dist), 0.f, 1.f);
}

void init_map()
{
    for(U32 x = 0; x < map_size.x; ++x)
    {
        for(U32 y = 0; y < map_size.y; ++y)
        {
            F32 o1 = simplex(vec2(x, y) * 0.005f) * 1.f;
            F32 o2 = simplex(vec2(x, y) * 0.01f) * 0.5f;
            F32 o3 = simplex(vec2(x, y) * 0.02f) * 0.25f;
            F32 o4 = simplex(vec2(x, y) * 0.04f) * 0.125f;
            F32 o5 = simplex(vec2(x, y) * 0.08f) * 0.0625f;
            F32 o6 = simplex(vec2(x, y) * 0.16f) * 0.03125f;
            F32 o7 = simplex(vec2(x, y) * 0.32f) * 0.01575f;
            F32 o8 = simplex(vec2(x, y) * 0.64f) * 0.007875f;
            F32 h = ((((o1 + o2 + o3 + o4 + o5 + o6 + o7 + o8) / 1.993235f) + 1.f) / 2.f) * map_size.z;
            for(U32 z = 0; z < h; ++z)
            {
                F32 rel_h = ((F32)h / (F32)map_size.z);
                Voxel voxel;
                     if(rel_h > 0.8)  voxel = snow_voxel;
                else if(rel_h > 0.5)  voxel = stone_voxel;
                else if(rel_h > 0.4)  voxel = dirt_voxel;
                else if(rel_h > 0.33) voxel = grass_voxel;
                else                  voxel = sand_voxel;
                voxel.col.r = blur(voxel.col.r, 16.f);
                voxel.col.g = blur(voxel.col.g, 16.f);
                voxel.col.b = blur(voxel.col.b, 16.f);
                voxel.col   *= randf(0.75f, 1.f);
                map[get_map_idx({x, y, z})] = voxel;
            }
            for(U32 z = h; z < map_size.z; ++z)
            {
                F32 rel_z = ((F32)z / (F32)map_size.z);
                F32 f = simplex(vec3(x, y, z) * 0.07f);
                Voxel voxel;
                if(rel_z < 0.3f) voxel = water_voxel;
                else
                {
                    if(f > 1.f - rel_z)
                    {
                        voxel = fog_voxel;
                        voxel.col.r *= randf(0.5, 1.f);
                        voxel.col.g *= randf(0.5, 1.f);
                        voxel.col.b *= randf(0.5, 1.f);
                    }
                    else                voxel = empty_voxel;
                }
                map[get_map_idx({x, y, z})] = voxel;
            }
        }
    }
}

void init_fb()
{
    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);

    GLuint fb_id;
    glGenFramebuffers(1, &fb_id);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb_id);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, id, 0);
}

void render_screen()
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, win_size.x, win_size.y,
                 0, GL_RGB, GL_UNSIGNED_BYTE, cast_plane);
    glBlitFramebuffer(0, 0, win_size.x            , win_size.y,
                      0, 0, win_size.x * win_scale, win_size.y * win_scale,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void check_gl_error()
{
    GLenum error = glGetError();
    if(error != GL_NO_ERROR)
    {
        std::string str;
        switch(error)
        {
            case GL_INVALID_ENUM: str = "invalid enum"; break;
            case GL_INVALID_VALUE: str = "invalid value"; break;
            case GL_INVALID_OPERATION: str = "invalid operation"; break;
            case GL_OUT_OF_MEMORY: str = "out of memory"; break;
            default: str = "unknown"; break;
        }
        throw std::runtime_error(str);
    }
}

struct RayHit
{
    vec4 col;
    vec3 pos;
    vec3 vox_pos;
    vec3 norm;
};

U32 i = 0;

bool cast_ray(vec3 const &from, vec3 const &to, RayHit &hit)
{
    ivec3 iter = (ivec3)from;
    vec3 ray = to - from;
    ivec3 r_sign = (ivec3)sign(ray);
    vec3 t_delta;
    vec3 t_max;
    vec3 dist;

    t_delta.x = (r_sign.x != 0) ? (F32)r_sign.x / ray.x : FLT_MAX;
    t_delta.y = (r_sign.y != 0) ? (F32)r_sign.y / ray.y : FLT_MAX;
    t_delta.z = (r_sign.z != 0) ? (F32)r_sign.z / ray.z : FLT_MAX;
    dist.x = r_sign.x > 0 ? (1.f - from.x + iter.x) :
                            (from.x - iter.x);
    dist.y = r_sign.y > 0 ? (1.f - from.y + iter.y) :
                            (from.y - iter.y);
    dist.z = r_sign.z > 0 ? (1.f - from.z + iter.z) :
                            (from.z - iter.z);
    t_max.x = r_sign.x != 0 ? t_delta.x * dist.x : FLT_MAX;
    t_max.y = r_sign.y != 0 ? t_delta.y * dist.y : FLT_MAX;
    t_max.z = r_sign.z != 0 ? t_delta.z * dist.z : FLT_MAX;

    F32 t = 0.f;
    hit.norm = {0, 0, 0};

    vec3 col = {0, 0, 0};
    enum { X, Y, Z } normal;
    F32  alpha = 0;
    Voxel const *vox;
    while(t_max.x <= 1.f || t_max.y <= 1.f || t_max.z <= 1.f)
    {
        if(t_max.x < t_max.y)
        {
            if(t_max.x < t_max.z)
            {
                iter.x  += r_sign.x;
                t_max.x += t_delta.x;
                t = t_max.x;
                normal = X;
            }
            else
            {
                iter.z  += r_sign.z;
                t_max.z += t_delta.z;
                t = t_max.z;
                normal = Z;
            }
        }
        else
        {
            if(t_max.y < t_max.z)
            {
                iter.y  += r_sign.y;
                t_max.y += t_delta.y;
                t = t_max.y;
                normal = Y;
            }
            else
            {
                iter.z  += r_sign.z;
                t_max.z += t_delta.z;
                t = t_max.z;
                normal = Z;
            }
        }
        vox    = &get_voxel(iter);
        col   += vox->col * std::min(1.f - alpha, vox->a);
        alpha += std::min(1.f - alpha, vox->a);
        if(alpha >= 1.f)
        {
                 if(normal == X) hit.norm = {-r_sign.x, 0, 0};
            else if(normal == Y) hit.norm = {0, -r_sign.y, 0};
            else if(normal == Z) hit.norm = {0, 0, -r_sign.z};
            i++;
            hit.col = vec4(col / alpha, 1.f);
            hit.pos = from + t * ray;
            hit.vox_pos = iter;
            return true;
        }
    }
    return false;
}

void cast()
{
    F32 z_near = 0.01f;
    F32 z_far  = -512.f;
    vec3 origin = player_pos + vec3(0, 0, 2);
    mat4 rot(1.f);
    rot = translate(rot, origin);
    rot *= eulerAngleZY(player_yaw, player_pitch);
    auto plane = [&] (F32 x, vec2 const &pos) -> vec3
    {
        return vec3(rot * vec4(x, pos, 1.f));
    };
    vec2 win_half = -(vec2)(win_size / 2u);
    vec3 near_pixel = plane(z_near, win_half / (vec2)win_size);
    vec3 near_delta_x = plane(z_near, (win_half + vec2(1, 0)) / (vec2)win_size)
                        - near_pixel;
    vec3 near_delta_y = plane(z_near, (win_half + vec2(0, 1)) / (vec2)win_size)
                        - near_pixel;
    vec3 far_pixel = plane(z_far, win_half * 16.f);
    vec3 far_delta_x = plane(z_far, (win_half + vec2(1, 0)) * 16.f) - far_pixel;
    vec3 far_delta_y = plane(z_far, (win_half + vec2(0, 1)) * 16.f) - far_pixel;

    RayHit hit;
    for(U32 i = 0; i < win_volume; ++i)
    {
        vec2 plane_pos = {(i % win_size.x) + 0.5f, (i / win_size.x) + 0.5f};
        vec3 near = near_pixel +
                    near_delta_x * (plane_pos.x) +
                    near_delta_y * (plane_pos.y);
        vec3 far = far_pixel +
                   far_delta_x * (plane_pos.x) +
                   far_delta_y * (plane_pos.y);
        if(cast_ray(near, far, hit))
        {
            if((ivec3)(hit.vox_pos + hit.norm) == player_cursor)
            {
                hit.col = {1.f, 1.f, 0.f, 1.f};
            }
            cast_plane[i] = vec3(hit.col) * hit.col.a * 255.f;
        }
        else cast_plane[i] = empty_voxel.col * 255.f;
    }
}

int main()
{
    std::srand(std::time(NULL));
    init_gl();
    init_map();
    init_fb();

    util::TickClock clock(util::TickClock::Duration(1.f) / 30.f);

    dvec2 mouse_pos;
    dvec2 last_mouse_pos(0, 0);
    U32 place_delay = 0;
    U32 dest_delay = 0;
    U32 jump_delay = 0;
    while(!glfwWindowShouldClose(win))
    {
        clock.start();
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);

        cast();
        render_screen();
        check_gl_error();

        glfwGetCursorPos(win, &mouse_pos.x, &mouse_pos.y);
        player_yaw   += (F32)(last_mouse_pos.x - mouse_pos.x) * 0.005f;
        player_pitch += (F32)(last_mouse_pos.y - mouse_pos.y) * 0.005f;
        player_pitch  = std::clamp(player_pitch, -(TAU / 4.f) + .001f,
                                                  (TAU / 4.f) - .001f);

        last_mouse_pos = mouse_pos;

        vec2 dir = {0, 0};
        if(glfwGetKey(win, GLFW_KEY_W)) dir.x = -1.f;
        if(glfwGetKey(win, GLFW_KEY_S)) dir.x =  1.f;
        if(glfwGetKey(win, GLFW_KEY_A)) dir.y = -1.f;
        if(glfwGetKey(win, GLFW_KEY_D)) dir.y =  1.f;
        if(glfwGetKey(win, GLFW_KEY_SPACE) &&
           get_voxel((ivec3)(player_pos - vec3(0, 0, 0.3))).a >= 1.f)
        {
            if(jump_delay == 0)
            {
                jump_delay = 10;
                player_vel.z = 1.0f;
            }
        }
        if(length(dir) > 0)
        {
            mat4 rot = eulerAngleZ(player_yaw);
            dir = vec2(rot * vec4(normalize(dir), 0.f, 1.f));
            player_vel.x = dir.x * 0.2f;
            player_vel.y = dir.y * 0.2f;
        }
        player_vel += vec3(0, 0, -0.1f);

        RayHit hit;
        if(!cast_ray(player_pos, player_pos + vec3(player_vel.x, 0, 0), hit))
        {
            player_pos.x += player_vel.x;
        }
        else player_vel.x = 0;
        if(!cast_ray(player_pos, player_pos + vec3(0, player_vel.y, 0), hit))
        {
            player_pos.y += player_vel.y;
        }
        else player_vel.y = 0;
        if(!cast_ray(player_pos, player_pos + vec3(0, 0, player_vel.z), hit))
        {
            player_pos.z += player_vel.z;
        }
        else player_vel.z = 0;
        player_vel *= 0.8f;

        F32 range = 10.f;
        mat4 rot = eulerAngleZY(player_yaw, player_pitch);
        vec3 dest = vec3(rot * vec4(-1.f, 0.f, 0.f, 1.f));
        if(cast_ray(player_pos + vec3(0, 0, 2.5),
                    player_pos + vec3(0, 0, 2.5) + dest * range, hit))
        {
            I32 state = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT);
            if(state == GLFW_PRESS && place_delay == 0)
            {
                place_delay = 10;
                if(!outside_bounds((ivec3)(hit.vox_pos + hit.norm)))
                {
                    Voxel vox = player_voxel;
                    vox.col.r = blur(vox.col.r, 4.f);
                    vox.col.g = blur(vox.col.g, 4.f);
                    vox.col.b = blur(vox.col.b, 4.f);
                    map[get_map_idx((uvec3)(hit.vox_pos + hit.norm))] = vox;
                }
            }
            state = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT);
            if(state == GLFW_PRESS && dest_delay == 0)
            {
                dest_delay = 3;
                if(!outside_bounds((ivec3)hit.vox_pos))
                {
                    map[get_map_idx((uvec3)hit.vox_pos)] = empty_voxel;
                }
            }
            else
            {
                player_cursor = (ivec3)(hit.vox_pos + hit.norm);
            }
        }
        else player_cursor = {-1, -1, -1};
        if(place_delay > 0) place_delay--;
        if(dest_delay > 0) dest_delay--;
        if(jump_delay > 0) jump_delay--;

        glfwSwapBuffers(win);
        glfwPollEvents();
        clock.stop();
        clock.synchronize();
    }

    glfwTerminate();
    return 0;
}

