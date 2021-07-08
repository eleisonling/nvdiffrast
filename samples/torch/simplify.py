import argparse
import os
import imageio
import pathlib
import numpy as np
import pywavefront
import torch
import nvdiffrast.torch as dr
import util


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def render(glctx, mtx, pos, pos_idx, vtx_col, resolution: int):
    pos_clip = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _ = dr.interpolate(vtx_col[None, ...], rast_out, pos_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color


def make_grid(arr, ncols=2):
    n, height, width, nc = arr.shape
    nrows = n // ncols
    assert n == nrows * ncols
    return arr.reshape(nrows, ncols, height, width, nc).swapaxes(1, 2).reshape(height * nrows, width * ncols, nc)


def simplification(max_iter=5000, resolution=4, discontinuous=False, repeats=1, log_interval=10, display_interval=None,
                   display_res=512, out_dir=None, log_fn=None, mp4save_interval=None, mp4save_fn=None):
    log_file = None
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(f'{out_dir}/{log_fn}', 'wt')
        if mp4save_interval != 0:
            writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', mode='I', fps=30, codec='libx264', bitrate='16M')
    else:
        mp4save_interval = None

    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    fn = "30501_LianPo_Mid.obj"
    refer_mesh = pywavefront.Wavefront(f'{datadir}/{fn}', collect_faces=True, create_materials=True)

    pos_idx = None
    vtx_pos = None
    vtx_tex = None
    vtx_normal = None
    vtx_color = None

    for mesh in refer_mesh.mesh_list:
        face = torch.tensor(mesh.faces, dtype=torch.int32)
        pos_idx = face if pos_idx is None else torch.cat((pos_idx, face))

        # tricky, assume only one material and with T2_N3_V3
        tex = torch.tensor([mesh.materials[0].vertices[::8], mesh.materials[0].vertices[1::8]])
        tex = tex.transpose(0, 1)
        vtx_tex = tex if vtx_tex is None else torch.cat((vtx_tex, tex))

        normal = torch.tensor([mesh.materials[0].vertices[2::8], mesh.materials[0].vertices[3::8],
                               mesh.materials[0].vertices[4::8]])
        normal = normal.T
        vtx_normal = normal if vtx_normal is None else torch.cat((vtx_normal, normal))

        vertex = torch.tensor([mesh.materials[0].vertices[5::8], mesh.materials[0].vertices[6::8],
                               mesh.materials[0].vertices[7::8]])
        vertex = vertex.T
        vtx_pos = vertex if vtx_pos is None else torch.cat((vtx_pos, vertex))

    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtx_pos.shape[0]))
    pos_idx = pos_idx.cuda()
    vtx_pos = vtx_pos.cuda()
    vtx_normal = vtx_normal.cuda()
    vtx_color = torch.ones(vtx_pos.shape[0], 3).cuda()

    glctx = dr.RasterizeGLContext()

    for rep in range(repeats):

        ang = 0.0
        gl_avg = []

        optimizer = torch.optim.Adam([vtx_color], lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10 ** (-x * 0.0005)))

        for epoch in range(max_iter + 1):
            r_rot = util.random_rotation_translation(0.25)
            a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))

            proj = util.projection(x=0.4, n=1., f=1000.0)
            r_mv = np.matmul(util.translate(0, 0, -500.5), r_rot)
            r_mvp = np.matmul(proj, r_mv).astype(np.float32)
            a_mv = np.matmul(util.translate(0, 0, -500.5), a_rot)
            a_mvp = np.matmul(proj, a_mv).astype(np.float32)

            # show/same image
            display_image = display_interval and (epoch % display_interval == 0)
            save_mp4 = mp4save_interval and (epoch % mp4save_interval == 0)

            if display_image or save_mp4:
                ang = ang + 0.01
                img_r = render(glctx, a_mvp, vtx_pos, pos_idx, vtx_color, display_res)[0]

                result_image = make_grid(np.stack([img_r.detach().cpu().numpy(), img_r.detach().cpu().numpy()]))

                if display_image:
                    util.display_image(result_image, size=display_res, title=f"{epoch} / {max_iter}")

    if writer is not None:
        writer.close()

    if log_file:
        log_file.close()


def main():
    parser = argparse.ArgumentParser(description='Simplify fit example')
    parser.add_argument('--outdir', help='Specify output directory', default='')
    parser.add_argument('--discontinuous', action='store_true', default=False)
    parser.add_argument('--resolution', type=int, default=0, required=True)
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save-interval', type=int, default=100)
    parser.add_argument('--max-iter', type=int, default=10000)
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        ds = 'd' if args.discontinuous else 'c'
        out_dir = f'{args.outdir}/cube_{ds}_{args.resolution}'
        print(f'Saving results under {out_dir}')
    else:
        out_dir = None
        print('No output directory specified, not saving log or images')

    simplification(max_iter=args.max_iter, resolution=args.resolution, discontinuous=args.discontinuous,
                   log_interval=10, display_interval=args.display_interval, out_dir=out_dir,
                   log_fn='log.txt', mp4save_interval=args.mp4save_interval, mp4save_fn='progress.mp4')

    print("Done")


if __name__ == "__main__":
    main()
