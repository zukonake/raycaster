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

struct Vert { vec2 pos; vec2 tx_pos; };

constexpr U32   win_scale = 5;
constexpr uvec2 win_size = {320, 180};
constexpr U32   win_volume = win_size.x * win_size.y;
constexpr uvec3 map_size = {512, 512, 256};
constexpr U32   map_volume = map_size.x * map_size.y * map_size.z;

constexpr F32 TAU = pi<F32>() * 2.f;

struct Voxel
{
    vec3 col;
    F32  a;
};

constexpr Voxel border_voxel = {{0.0f, 0.0f, 0.0f}, 1.0f};
constexpr Voxel empty_voxel  = {{0.0f, 0.0f, 0.0f}, 0.01f};
constexpr Voxel water_voxel  = {{0.2f, 0.5f, 0.8f}, 0.1f};
constexpr Voxel sand_voxel   = {{1.0f, 1.0f, 0.6f}, 1.0f};
constexpr Voxel stone_voxel  = {{0.4f, 0.4f, 0.4f}, 1.0f};
constexpr Voxel snow_voxel   = {{0.9f, 0.9f, 1.0f}, 1.0f};
constexpr Voxel grass_voxel  = {{0.3f, 0.8f, 0.2f}, 1.0f};
constexpr Voxel dirt_voxel   = {{0.6f, 0.3f, 0.1f}, 1.0f};

Voxel map[map_volume];
U8    cast_plane[win_volume * 3];
GLFWwindow *win;

struct OctNode
{
    union
    {
        OctNode *nodes[8];
        Voxel    leaf;
    };
    bool is_leaf;
};

F32  player_yaw = TAU / 3.f;
F32  player_pitch = 0;
vec3 player_vel = {0, 0, 0};
vec3 player_pos = {map_size.x / 2, map_size.y / 2, 250};

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
    return pos.x + pos.y * map_size.x + pos.z * map_size.x * map_size.y;
}

Voxel const &get_voxel(ivec3 const &pos)
{
    if(outside_bounds(pos)) return border_voxel;
    else return map[get_map_idx((uvec3)pos)];
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
    glfwSwapInterval(0);

    if(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress) == 0)
    {
        throw std::runtime_error("GLAD error");
    }

    glViewport(0, 0, win_size.x * win_scale, win_size.y * win_scale);
}

F32 randf(F32 dist)
{
    return (((F32)(std::rand() % 0x10000) / 0x10000) / dist) +
           ((1.f - (1.f / dist)) / 2.f);
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
                voxel.col *= randf(8.f);
                map[get_map_idx({x, y, z})] = voxel;
            }
            for(U32 z = h; z < map_size.z; ++z)
            {
                F32 rel_z = ((F32)z / (F32)map_size.z);
                if(rel_z < 0.3) map[get_map_idx({x, y, z})] = water_voxel;
                else            map[get_map_idx({x, y, z})] = empty_voxel;
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

vec4 cast_ray(vec3 const &from, vec3 const &to)
{
    ivec3 iter = (ivec3)from;
    vec3 ray = to - from;
    ivec3 r_sign = (ivec3)sign(ray);
    vec3 t_delta;
    vec3 t_max;
    t_delta.x = (r_sign.x != 0) ? (F32)r_sign.x / ray.x : FLT_MAX;
    t_delta.y = (r_sign.y != 0) ? (F32)r_sign.y / ray.y : FLT_MAX;
    t_delta.z = (r_sign.z != 0) ? (F32)r_sign.z / ray.z : FLT_MAX;
    t_max.x = r_sign.x > 0 ? t_delta.x * (1.f - from.x + std::floor(from.x)) :
                             t_delta.x * (from.x - std::floor(from.x));
    t_max.y = r_sign.y > 0 ? t_delta.y * (1.f - from.y + std::floor(from.y)) :
                             t_delta.y * (from.y - std::floor(from.y));
    t_max.z = r_sign.z > 0 ? t_delta.z * (1.f - from.z + std::floor(from.z)) :
                             t_delta.z * (from.z - std::floor(from.z));

    vec3 col = {0, 0, 0};
    F32  alpha = 0;
    Voxel const *vox = &get_voxel(iter);
    col   += vox->col * std::min(1.f - alpha, vox->a);
    alpha += vox->a;
    if(alpha >= 1.f) return vec4(col / alpha, 1.f);
    while(true)
    {
        if(t_max.x < t_max.y)
        {
            if(t_max.x < t_max.z)
            {
                iter.x  += r_sign.x;
                t_max.x += t_delta.x;
            }
            else
            {
                iter.z  += r_sign.z;
                t_max.z += t_delta.z;
            }
        }
        else
        {
            if(t_max.y < t_max.z)
            {
                iter.y  += r_sign.y;
                t_max.y += t_delta.y;
            }
            else
            {
                iter.z  += r_sign.z;
                t_max.z += t_delta.z;
            }
        }
        vox = &get_voxel(iter);
        col   += vox->col * std::min(1.f - alpha, vox->a);
        alpha += vox->a;
        if(alpha >= 1.f) return vec4(col / alpha, 1.f);
        if(t_max.x > 1.f && t_max.y > 1.f && t_max.z > 1.f) break;
    }
    return {empty_voxel.col, empty_voxel.a};
}

void cast()
{
    F32 z_near = 0.01f;
    F32 z_far  = -256.f;
    vec3 origin = player_pos + vec3(0, 0, 1);
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
    vec3 far_pixel = plane(z_far, win_half * 2.f);
    vec3 far_delta_x = plane(z_far, (win_half + vec2(1, 0)) * 2.f) - far_pixel;
    vec3 far_delta_y = plane(z_far, (win_half + vec2(0, 1)) * 2.f) - far_pixel;
    for(U32 i = 0; i < win_volume; ++i)
    {
        vec2 plane_pos = {(i % win_size.x) + 0.5f, (i / win_size.x) + 0.5f};
        vec3 near = near_pixel +
                    near_delta_x * plane_pos.x +
                    near_delta_y * plane_pos.y;
        vec3 far = far_pixel +
                   far_delta_x * plane_pos.x +
                   far_delta_y * plane_pos.y;
        vec4 col = cast_ray(near, far);
        cast_plane[i * 3 + 0] = col.r * 255.f * col.a;
        cast_plane[i * 3 + 1] = col.g * 255.f * col.a;
        cast_plane[i * 3 + 2] = col.b * 255.f * col.a;
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

        vec3 dir = {0, 0, 0};
        if(glfwGetKey(win, GLFW_KEY_W)) dir.x = -1.f;
        if(glfwGetKey(win, GLFW_KEY_S)) dir.x =  1.f;
        if(glfwGetKey(win, GLFW_KEY_A)) dir.y = -1.f;
        if(glfwGetKey(win, GLFW_KEY_D)) dir.y =  1.f;
        if(glfwGetKey(win, GLFW_KEY_SPACE) &&
           get_voxel((ivec3)(player_pos - vec3(0, 0, 1))).a >= 1.f)
        {
            dir.z = 10.f;
        }
        if(dir != vec3(0, 0, 0))
        {
            mat4 rot = eulerAngleZ(player_yaw);
            dir = vec3(rot * vec4(dir, 1.f));
            player_vel += dir * 0.3f;
        }
        player_vel += vec3(0, 0, -0.3f);

        if(cast_ray(player_pos, player_pos + vec3(player_vel.x, 0, 0)).a < 1.f)
        {
            player_pos.x += player_vel.x;
        }
        else player_vel.x = 0;
        if(cast_ray(player_pos, player_pos + vec3(0, player_vel.y, 0)).a < 1.f)
        {
            player_pos.y += player_vel.y;
        }
        else player_vel.y = 0;
        if(cast_ray(player_pos, player_pos + vec3(0, 0, player_vel.z)).a < 1.f)
        {
            player_pos.z += player_vel.z;
        }
        else player_vel.z = 0;
        player_vel *= 0.5f;

        glfwSwapBuffers(win);
        glfwPollEvents();
        clock.stop();
        clock.synchronize();
    }

    glfwTerminate();
    return 0;
}

