#include <cstdlib>
#include <cstdint>
#include <cfloat>
#include <ctime>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
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
#include <CL/cl.h>
//
#include "tick_clock.hpp"

using namespace glm;

typedef int8_t   I8;
typedef int16_t  I16;
typedef int32_t  I32;
typedef int64_t  I64;

typedef uint8_t  U8;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;

typedef float    F32;
typedef double   F64;

constexpr F32   fps        = 30.f;
constexpr U32   win_scale  = 1;
constexpr uvec2 win_size   = {1600, 900};
constexpr U32   win_volume = win_size.x * win_size.y;

constexpr F32 TAU = pi<F32>() * 2.f;

struct Voxel
{
    vec3 col;
    F32  a;
    vec3 variance = {0, 0, 0};
};

constexpr Voxel border_voxel = {{0.7f, 0.7f, 0.8f}, 1.0f};
constexpr Voxel empty_voxel  = {{0.3f, 0.4f, 0.7f}, 0.004f};
constexpr Voxel fog_voxel    = {{0.8f, 0.8f, 0.9f}, 0.02f};
constexpr Voxel water_voxel  = {{0.1f, 0.2f, 0.5f}, 0.05f};
constexpr Voxel sand_voxel   = {{1.0f, 1.0f, 0.6f}, 1.0f};
constexpr Voxel stone_voxel  = {{0.4f, 0.4f, 0.4f}, 1.0f};
constexpr Voxel snow_voxel   = {{0.9f, 0.9f, 1.0f}, 1.0f};
constexpr Voxel grass_voxel  = {{0.3f, 0.8f, 0.2f}, 1.0f};
constexpr Voxel dirt_voxel   = {{0.6f, 0.3f, 0.1f}, 1.0f};
constexpr Voxel player_voxel = {{1.0f, 0.0f, 1.0f}, 1.0f};

tvec4<U8>   cast_plane[win_volume];
GLFWwindow *win;

cl_context       context;
cl_program       program;
cl_kernel        kernel;
cl_command_queue command_queue;
cl_mem           cl_screen;

struct Player
{
    F32   yaw = TAU / 3.f;
    F32   pitch = 0;
    ivec3 cursor = {-1, -1, -1};
    vec3  vel = {0, 0, 0};
    vec3  pos = {0, 0, 60};
    U32   place_delay = 0;
    U32   dest_delay = 0;
    U32   jump_delay = 0;
    dvec2 last_mouse_pos = (dvec2)(win_size * win_scale) / 2.0;
} player;

constexpr U64   chk_size_exp  = 4;
constexpr uvec3 chk_size      = {1 << chk_size_exp,
                                 1 << chk_size_exp,
                                 1 << chk_size_exp};
constexpr U32   chk_vol       = chk_size.x * chk_size.y * chk_size.z;
constexpr uvec3 chk_idx_mask  = {chk_size.x - 1,
                                 chk_size.y - 1,
                                 chk_size.z - 1};
constexpr uvec3 chk_idx_shift = {0,
                                 chk_size_exp,
                                 chk_size_exp + chk_size_exp};
constexpr uvec3 chk_pos_shift = uvec3(chk_size_exp);
constexpr U32   chunk_unload_time = 100;

struct Identity
{
    std::size_t operator()(U64 const &v) const { return v; }
};

struct Chunk
{
    Voxel voxels[chk_vol];
    U32   unload_timer;
};

U8 chunk_io_buff[chk_vol * 5];

constexpr F32   max_map_h = 128;
std::unordered_map<U64, Chunk, Identity> map;
std::unordered_map<U64, F32[chk_vol], Identity> height_map;

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

U64 pack_vec(ivec3 const &v)
{
    return  (v.x & 0xFFFFFul) |
           ((v.y & 0xFFFFFul) << 20ul) |
           ((v.z & 0xFFFFFul) << 40ul);
}

U64 get_chk_idx(ivec3 const &pos)
{
    uvec3 idx_pos = ((uvec3 const &)pos & chk_idx_mask) << chk_idx_shift;
    return idx_pos.x | idx_pos.y | idx_pos.z;
}

ivec3 get_chk_pos(ivec3 const &pos)
{
    return (ivec3)((uvec3 const &)pos >> chk_pos_shift);
}

ivec3 get_map_pos(ivec3 const &chk_pos)
{
    return (ivec3)((uvec3 const &)chk_pos << chk_pos_shift);
}

std::string get_chunk_path(U64 idx)
{
    return std::string("chunks/" + std::to_string(idx));
}

void encode_chunk(U8 *buff, Chunk const &chunk)
{
    U16 alpha;
    for(U32 i = 0; i < chk_vol; ++i)
    {
        buff[i * 5 + 0] = (U8)(chunk.voxels[i].col.r * 255.f);
        buff[i * 5 + 1] = (U8)(chunk.voxels[i].col.g * 255.f);
        buff[i * 5 + 2] = (U8)(chunk.voxels[i].col.b * 255.f);
        alpha = (U16)(std::round(chunk.voxels[i].a * 65535.f));
        buff[i * 5 + 3] = ((U8 *)&alpha)[0];
        buff[i * 5 + 4] = ((U8 *)&alpha)[1];
    }
}

void decode_chunk(U8 const *buff, Chunk &chunk)
{
    U16 alpha;
    for(U32 i = 0; i < chk_vol; ++i)
    {
        chunk.voxels[i].col.r = (F32)buff[i * 5 + 0] / 255.f;
        chunk.voxels[i].col.g = (F32)buff[i * 5 + 1] / 255.f;
        chunk.voxels[i].col.b = (F32)buff[i * 5 + 2] / 255.f;
        ((U8 *)&alpha)[0] = buff[i * 5 + 3];
        ((U8 *)&alpha)[1] = buff[i * 5 + 4];
        chunk.voxels[i].a = (F32)alpha / 65535.f;
    }
}

void save_chunk(U64 idx, Chunk const &chunk)
{
    std::ofstream f;
    f.open(get_chunk_path(idx), std::ios::binary);
    if(!f.good())
    {
        throw std::runtime_error("couldn't save chunk "
                                 "you need to create the folder \"chunks\"");
    }
    encode_chunk(chunk_io_buff, chunk);
    f.write((char *)chunk_io_buff, 5 * chk_vol);
    f.close();
}

bool load_chunk(U64 idx, Chunk &chunk)
{
    std::ifstream f;
    f.open(get_chunk_path(idx), std::ios::binary);
    if(f.good())
    {
        f.read((char *)chunk_io_buff, 5 * chk_vol);
        decode_chunk(chunk_io_buff, chunk);
        f.close();
        return true;
    }
    else return false;
}

void unload_chunks()
{
    for(auto it = map.begin(); it != map.end();)
    {
        --it->second.unload_timer;
        if(it->second.unload_timer == 0)
        {
            save_chunk(it->first, it->second);
            it = map.erase(it);
        }
        else ++it;
    }
}

void save_all()
{
    for(auto const &it : map)
    {
        save_chunk(it.first, it.second);
    }
}

void generate_height_chunk(ivec3 const &chk_h_pos)
{
    constexpr U32 octaves = 8;
    constexpr F32 base_scale = 0.0025f;
    constexpr F32 height_exp = 2.f;
    U64 packed_chk_h_pos = pack_vec(chk_h_pos);
    height_map[packed_chk_h_pos];
    for(U32 y = 0; y < chk_size.y; ++y)
    {
        for(U32 x = 0; x < chk_size.x; ++x)
        {
            vec2 map_h_pos = vec2(get_map_pos(chk_h_pos) + ivec3(x, y, 0));
            F32 scale  = base_scale;
            F32 weight = 1.f;
            F32 avg    = 0.f;
            for(U32 o = 0; o < octaves; ++o)
            {
                avg    += simplex(map_h_pos * scale) * weight;
                scale  *= 2.f;
                weight /= 2.f;
            }
            F32 h = std::pow(((avg / (2.f - (weight * 2.f)))
                             + 1.f) / 2.f, height_exp) * max_map_h;
            height_map[packed_chk_h_pos][get_chk_idx({x, y, 0})] = h;
        }
    }
}

void generate_chunk(Chunk &chunk, ivec3 const &chk_pos)
{
    U64 packed_chk_idx   = pack_vec(chk_pos);
    if(load_chunk(packed_chk_idx, chunk)) return;
    ivec3 chk_h_pos = {chk_pos.x, chk_pos.y, 0};
    U64 packed_chk_h_pos = pack_vec(chk_h_pos);
    if(height_map.count(packed_chk_h_pos) == 0)
    {
        generate_height_chunk(chk_h_pos);
    }
    for(U32 z = 0; z < chk_size.z; ++z)
    {
        for(U32 y = 0; y < chk_size.y; ++y)
        {
            for(U32 x = 0; x < chk_size.x; ++x)
            {
                vec3 map_pos = (vec3)(get_map_pos(chk_pos) + ivec3(x, y, z));

                F32 h = height_map[pack_vec(chk_h_pos)][get_chk_idx({x, y, 0})];
                
                Voxel voxel;
                if(map_pos.z <= h)
                {
                    F32 rel_h = h / max_map_h;
                         if(rel_h > 0.8)  voxel = snow_voxel;
                    else if(rel_h > 0.5)  voxel = stone_voxel;
                    else if(rel_h > 0.4)  voxel = dirt_voxel;
                    else if(rel_h > 0.33) voxel = grass_voxel;
                    else                  voxel = sand_voxel;
                    voxel.col.r = blur(voxel.col.r, 32.f);
                    voxel.col.g = blur(voxel.col.g, 32.f);
                    voxel.col.b = blur(voxel.col.b, 32.f);
                    voxel.col   *= randf(0.75f, 1.f);
                    voxel.col   *= std::clamp((F32)map_pos.z / h, 0.f, 1.f);
                }
                else
                {
                    F32 rel_z = (F32)map_pos.z / max_map_h;
                    if(rel_z < 0.3f) voxel = water_voxel;
                    else
                    {
                        voxel = empty_voxel;
                        if(rel_z > 0.4f)
                        {
                            F32 f = simplex(map_pos * 0.04f);
                            if(f > 1.4f - rel_z)
                            {
                                voxel = fog_voxel;
                                voxel.col.r *= randf(0.5, 1.f);
                                voxel.col.g *= randf(0.5, 1.f);
                                voxel.col.b *= randf(0.5, 1.f);
                            }
                        }
                    }
                }
                chunk.voxels[get_chk_idx({x, y, z})] = voxel;
            }
        }
    }
}

Chunk &get_chunk(ivec3 const &chk_pos)
{
    U64 idx = pack_vec(chk_pos);
    if(map.count(idx) == 0)
    {
        map[idx];
        generate_chunk(map[idx], chk_pos);
    }
    map[idx].unload_timer = chunk_unload_time;
    return map[idx];
}

Voxel &get_vox(ivec3 const &pos)
{
    return get_chunk(get_chk_pos(pos)).voxels[get_chk_idx(pos)];
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

    GLuint tex_id;
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);

    GLuint fb_id;
    glGenFramebuffers(1, &fb_id);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb_id);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, tex_id, 0);
}

void check_cl_error(cl_int err)
{
    if(err != CL_SUCCESS)
    {
        throw std::runtime_error("OpenCL error: " + std::to_string(err));
    }
}

void init_cl()
{
    const char *program_path = "src/cast_ray.cl";

    cl_uint platform_id_count = 0;
    cl_int err = CL_SUCCESS;
    clGetPlatformIDs(0, nullptr, &platform_id_count);

    if(platform_id_count == 0)
    {
        throw std::runtime_error("No OpenCL platforms found");
    }

    std::vector<cl_platform_id> platform_ids(platform_id_count);
    clGetPlatformIDs(platform_id_count, platform_ids.data(), nullptr);

    cl_uint device_id_count = 0;
    clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, 0, nullptr,
        &device_id_count);

    if(device_id_count == 0)
    {
        throw std::runtime_error("No OpenCL devices found");
    }

    std::vector<cl_device_id> device_ids (device_id_count);
    clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_GPU, device_id_count,
        device_ids.data (), nullptr);

    const cl_context_properties context_properties[] =
    {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform_ids[0]),
        0, 0
    };

    context = clCreateContext(context_properties, device_id_count,
        device_ids.data(), nullptr, nullptr, &err);
    check_cl_error(err);
    std::ifstream f(program_path);
    if(!f.good())
    {
        throw std::runtime_error("Couldn't load OpenCL program");
    }
    f.seekg(0, std::ios::end);
    std::streampos len = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> buff(len);
    f.read(buff.data(), len);
    f.close();
    
    std::size_t src_len[1]  = { buff.size() };
    char const *src_data[1] = { buff.data() };

    program =
        clCreateProgramWithSource(context, 1, src_data, src_len, nullptr);


    clBuildProgram(program, device_id_count, device_ids.data(),
                   nullptr, nullptr, nullptr);
    {
        std::size_t log_size;
        clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_size);
        std::vector<char> log(log_size);

        clGetProgramBuildInfo(program, device_ids[0], CL_PROGRAM_BUILD_LOG,
                              log_size, log.data(), nullptr);

        std::cout << std::string(log.data()) << std::endl;
    }

    kernel = clCreateKernel(program, "cast_plane", &err);
    check_cl_error(err);

    static const cl_image_format format = {CL_RGBA, CL_UNORM_INT8};

    cl_screen = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format,
        win_size.x, win_size.y, 0, nullptr, &err);
    check_cl_error(err);

    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_screen);
    check_cl_error(err);

    command_queue = clCreateCommandQueue(context, device_ids[0],
        CL_QUEUE_PROFILING_ENABLE, &err);
    check_cl_error(err);
}

void render_screen()
{
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, win_size.x, win_size.y,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, cast_plane);
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

bool cast_ray(vec3 const &from, vec3 const &to, RayHit &hit)
{
    ivec3 iter = (ivec3)floor(from);
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
    ivec3 cached_chk_pos = get_chk_pos(iter);
    ivec3 current_chk_pos;
    Voxel const *vox;
    Chunk const *chk = &get_chunk(cached_chk_pos);
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
        current_chk_pos = get_chk_pos(iter);
        if(cached_chk_pos != current_chk_pos)
        {
            chk = &get_chunk(current_chk_pos);
            cached_chk_pos = current_chk_pos;
        }
        vox    = &chk->voxels[get_chk_idx(iter)];
        col   += vox->col * std::min(1.f - alpha, vox->a);
        alpha += std::min(1.f - alpha, vox->a);
        if(alpha >= 1.0f)
        {
                 if(normal == X) hit.norm = {-r_sign.x, 0, 0};
            else if(normal == Y) hit.norm = {0, -r_sign.y, 0};
            else if(normal == Z) hit.norm = {0, 0, -r_sign.z};
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
    F32 z_far  = -128.f;
    vec3 origin = player.pos + vec3(0, 0, 2);
    mat4 rot(1.f);
    rot = translate(rot, origin);
    rot *= eulerAngleZY(player.yaw, player.pitch);
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
    vec3 far_pixel = plane(z_far, win_half * 1.6f);
    vec3 far_delta_x = plane(z_far, (win_half + vec2(1, 0)) * 1.6f) - far_pixel;
    vec3 far_delta_y = plane(z_far, (win_half + vec2(0, 1)) * 1.6f) - far_pixel;

    cl_uint err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_float3), &near_pixel);
    check_cl_error(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_float3), &near_delta_x);
    check_cl_error(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_float3), &near_delta_y);
    check_cl_error(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_float3), &far_pixel);
    check_cl_error(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_float3), &far_delta_x);
    check_cl_error(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_float3), &far_delta_y);
    check_cl_error(err);

    constexpr std::size_t global_work_size [] = {win_size.x, win_size.y, 0};
    constexpr std::size_t local_work_size  [] = {16, 16, 0};
    cl_event event;
    clEnqueueNDRangeKernel(command_queue, kernel, 2,
        nullptr, global_work_size, local_work_size, 0, nullptr, &event);
    clWaitForEvents(1, &event);
    clFinish(command_queue);
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double nanoSeconds = time_end-time_start;
    printf("OpenCl Execution time is: %0.3f milliseconds \n",nanoSeconds / 1000000.0);

    cl_event event2;
    constexpr std::size_t img_origin[3] = {0, 0, 0};
    constexpr std::size_t region[3]     = {win_size.x, win_size.y, 1};
    err = clEnqueueReadImage(command_queue, cl_screen, CL_TRUE,
        img_origin, region, 0, 0, cast_plane, 0, nullptr, nullptr);
    check_cl_error(err);
}

void handle_input()
{
    RayHit hit;
    dvec2 mouse_pos;

    glfwGetCursorPos(win, &mouse_pos.x, &mouse_pos.y);
    player.yaw   += (F32)(player.last_mouse_pos.x - mouse_pos.x) * 0.005f;
    player.pitch += (F32)(player.last_mouse_pos.y - mouse_pos.y) * 0.005f;
    player.pitch  = std::clamp(player.pitch, -(TAU / 4.f) + .001f,
                                              (TAU / 4.f) - .001f);

    player.last_mouse_pos = mouse_pos;

    vec3 dir = {0, 0, 0};
    if(glfwGetKey(win, GLFW_KEY_W)) dir.x = -1.f;
    if(glfwGetKey(win, GLFW_KEY_S)) dir.x =  1.f;
    if(glfwGetKey(win, GLFW_KEY_A)) dir.y = -1.f;
    if(glfwGetKey(win, GLFW_KEY_D)) dir.y =  1.f;
    if(glfwGetKey(win, GLFW_KEY_SPACE) &&
       get_vox((ivec3)floor(player.pos - vec3(0, 0, 0.3))).a >= 1.f)
    {
        if(player.jump_delay == 0)
        {
            player.jump_delay = 10;
            player.vel.z = 1.0f;
        }
    }
    if(length(dir) > 0)
    {
        mat4 rot = eulerAngleZY(player.yaw, player.pitch);
        dir = vec3(rot * vec4(normalize(dir), 1.f));
        player.vel = dir * 0.8f;
    }

    F32 range = 10.f;
    mat4 rot = eulerAngleZY(player.yaw, player.pitch);
    vec3 dest = vec3(rot * vec4(-1.f, 0.f, 0.f, 1.f));
    if(cast_ray(player.pos + vec3(0, 0, 1.5),
                player.pos + vec3(0, 0, 1.5) + dest * range, hit))
    {
        I32 state = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT);
        if(state == GLFW_PRESS && player.place_delay == 0)
        {
            player.place_delay = 10;
            Voxel vox = player_voxel;
            vox.col.r = blur(vox.col.r, 4.f);
            vox.col.g = blur(vox.col.g, 4.f);
            vox.col.b = blur(vox.col.b, 4.f);
            get_vox((ivec3)(hit.vox_pos + hit.norm)) = vox;
        }
        state = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT);
        if(state == GLFW_PRESS && player.dest_delay == 0)
        {
            player.dest_delay = 3;
            get_vox((ivec3)hit.vox_pos) = empty_voxel;
        }
        else
        {
            player.cursor = (ivec3)(hit.vox_pos + hit.norm);
        }
    }
    else player.cursor = {-1, -1, -1};
    if(player.place_delay > 0) player.place_delay--;
    if(player.dest_delay  > 0) player.dest_delay--;
    if(player.jump_delay  > 0) player.jump_delay--;
}

void handle_player_physics()
{
    RayHit hit;
    //player.vel += vec3(0, 0, -0.1f);

    //if(!cast_ray(player.pos, player.pos + vec3(player.vel.x, 0, 0), hit))
    {
        player.pos.x += player.vel.x;
    }
    //else player.vel.x = 0;
    //if(!cast_ray(player.pos, player.pos + vec3(0, player.vel.y, 0), hit))
    {
        player.pos.y += player.vel.y;
    }
    //else player.vel.y = 0;
    //if(!cast_ray(player.pos, player.pos + vec3(0, 0, player.vel.z), hit))
    {
        player.pos.z += player.vel.z;
    }
    //else player.vel.z = 0;
    player.vel *= 0.8f;

}

int main()
{
    std::srand(std::time(NULL));
    init_gl();
    init_cl();

    util::TickClock clock(util::TickClock::Duration(1.f) / fps);

    while(!glfwWindowShouldClose(win))
    {
        clock.start();
        glClearColor(0.0, 0.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);

        cast();
        render_screen();
        check_gl_error();

        handle_input();
        handle_player_physics();

        unload_chunks();
        
        glfwSwapBuffers(win);
        glfwPollEvents();
        clock.stop();
        auto delta = clock.synchronize();
        std::cout << delta.count() << std::endl;
    }

    save_all();
    glfwTerminate();

    clReleaseMemObject(cl_screen);
    clReleaseCommandQueue(command_queue);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}

