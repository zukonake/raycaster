constant uint  chk_size_exp    = 4;
constant uint  chk_size        = 1 << chk_size_exp;
constant uint  chk_vol         = chk_size * chk_size * chk_size;
constant uint  chk_pos_shift   = chk_size_exp;
constant uint  chk_idx_mask    = chk_size - 1;
constant uint3 chk_idx_shift   = {0,
                                  chk_size_exp,
                                  chk_size_exp + chk_size_exp};

constant ulong map_bucket_size = 2048 * 3;

constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_CLAMP_TO_EDGE   |
    CLK_FILTER_NEAREST;

typedef struct
{
    float3 col;
    float  alpha;
} VoxelType;

typedef struct
{
    float3 col;
    float  a;
    ushort id;
} Voxel;

typedef struct
{
    Voxel  voxels[chk_vol];
    ushort bucket_idx;
} Chunk;

typedef struct
{
    ulong  chk_idx;
    ushort bucket_idx;
} BucketEntry;

typedef struct
{
    Chunk      *bucket[map_bucket_size];
    BucketEntry bucket_entries[map_bucket_size];
} Map;

ulong get_chk_idx(int3 pos)
{
    uint3  idx_pos = (*(uint3 *)&pos & chk_idx_mask) << chk_idx_shift;
    return idx_pos.x | idx_pos.y | idx_pos.z;
}

int3 get_chk_pos(int3 pos)
{
    return pos >> chk_pos_shift;
}

ushort trunc_chk_idx(ulong chk_idx)
{
    return 0; //TODO
}

Chunk const *get_chunk(ulong chk_idx, Map const *map)
{
    return NULL;//map->bucket_entries(
}

float4 cast_ray(float3 from, float3 to)
{
    float3 ray = to - from;
    float3 from_floor;
    float3 from_fract = modf(from, &from_floor);

    int3 iter   = convert_int3(from_floor);
    int3 r_sign = convert_int3(sign(ray));

    float3 t_delta = select(FLT_MAX, convert_float3(r_sign) / ray, r_sign != 0);
    float3 dist    = select(from_fract, 1.f - from + from_floor, r_sign > 0);
    float3 t_max   = select(FLT_MAX, t_delta * dist, r_sign != 0);

    float3 col   = 0.f;
    float  alpha = 0.f;
    float  norm_vox_a;

    //int3 cached_chk_pos = get_chk_pos(iter);
    //int3 current_chk_pos;

    Voxel vox;
    //Chunk const *chk = get_chunk(get_chk_idx(cached_chk_pos), NULL);

    for(uint i = 0; i < 1024; ++i)
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
        /*current_chk_pos = get_chk_pos(iter);
        if(any(cached_chk_pos != current_chk_pos))
        {
            chk = get_chunk(get_chk_idx(current_chk_pos), NULL);
            cached_chk_pos = current_chk_pos;
        }*/
        //vox    = &chk->voxels[get_chk_idx(iter)];
        vox.col = iter.z < 0 ? iter.x > 0 ? 1.f : 1.f : 0.f;
        vox.a   = iter.z < 0 ? iter.x > 0 ? 1.f : 1.f : 0.f;
        alpha   = vox.a;

        /*norm_vox_a = min(1.f - alpha, vox.a);
        col   += vox.col * norm_vox_a;
        alpha += norm_vox_a;*/

        if(alpha >= 1.f || all(t_max > 1.f)) break;
    }
    vox.col *= 1.f - fast_distance(from_floor, convert_float3(iter)) /
                     fast_distance(from, to);
    return (float4)(vox.col, 1.f);
}

float4 cast_ray2(float3 s, float3 d)
{
    Voxel vox;

    float3 r = d - s;
    float3 a = fabs(r);
    float  m = a.x > a.y && a.x > a.z ? a.x :
               a.y > a.x && a.y > a.z ? a.y : a.z;
    float3 dt = r / m;
    float3 s_floor;
    float3 f = fract(s, &s_floor);
    float  o = a.x > a.y && a.x > a.z ? r.x < 0 ? f.x : 1.f - f.x :
               a.y > a.x && a.y > a.z ? r.y < 0 ? f.y : 1.f - f.y :
               r.z < 0 ? f.z : 1.f - f.z;

    float3 it = s + o * dt;
    float alpha = 0.f;

    for(uint i = 0; i < 1024; ++i)
    {
        vox.col = it.z < 0.f ? 1.f : 0.f;
        vox.a   = it.z < 0.f ? 1.f : 0.f;
        alpha   = vox.a; 
        if(alpha >= 1.f) break;
        it += dt;
    }
    vox.col *= 1.f - fast_distance(floor(s), convert_float3(it)) /
                     fast_distance(floor(s), floor(d));
    return (float4)(vox.col, 1.f);
}

kernel void cast_plane(float3 near_pixel, float3 near_d_x, float3 near_d_y,
                       float3 far_pixel,  float3 far_d_x , float3 far_d_y,
                       write_only image2d_t screen)
{
    int2  group_id   = {get_group_id(0)  , get_group_id(1)};
    int2  local_id   = {get_local_id(0)  , get_local_id(1)};
    int2  local_size = {get_local_size(0), get_local_size(1)};
    int2  screen_pos = group_id * local_size + local_id;
    float2 f_id = convert_float2(screen_pos);

    float3 from = near_pixel + f_id.x * near_d_x + f_id.y * near_d_y;
    float3 to   = far_pixel  + f_id.x * far_d_x  + f_id.y * far_d_y;

    write_imagef(screen, screen_pos, cast_ray2(from, to));
}

