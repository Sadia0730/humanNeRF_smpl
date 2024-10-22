import torch
import torch.nn as nn
import torch.nn.functional as F
from third_parties.smpl.smpl_numpy import SMPL
from core.utils.network_util import MotionBasisComputer, check_for_nans
from core.nets.human_nerf.component_factory import \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp, \
    load_density_mlp

from configs import cfg
from core.nets.human_nerf.triplane.triplane import TriPlane

#from absl import flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('cfg',
#                     '387.yaml',
#                     'the path of config file')

MODEL_DIR = 'third_parties/smpl/models'



class Network_Triplane(nn.Module):
    def __init__(
            self,
            n_features=32,
            triplane_res=256,):
        super(Network_Triplane, self).__init__()
        #triplane features
        self.triplane = TriPlane(n_features, resX=triplane_res, resY=triplane_res, resZ=triplane_res).to('cuda')
        # Calculate triplane feature size
        tri_feats_size = 3 * n_features  # Since tri_feats is concatenated from 3 planes: xy, xz, yz
        self.blending_weight = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
                                        cfg.non_rigid_motion_mlp.i_embed)
        self.non_rigid_mlp = \
            load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                skips=cfg.non_rigid_motion_mlp.skips)
        self.non_rigid_mlp = \
            nn.DataParallel(
                self.non_rigid_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.secondary_gpus[0])

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn

        # canonical mlp 
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cnl_pos_embed_size,
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips)
        self.cnl_mlp = \
            nn.DataParallel(
                self.cnl_mlp,
                device_ids=cfg.secondary_gpus,
                output_device=cfg.primary_gpus[0])

        # Load density MLP and set to primary device first
        self.density_mlp = \
            load_density_mlp(cfg.density_mlp.module)(
            input_ch=tri_feats_size,
            mlp_depth=cfg.density_mlp.mlp_depth,
            mlp_width=cfg.density_mlp.mlp_width
        ).to(cfg.secondary_gpus[0])
        for name, param in self.density_mlp.named_parameters():
            print(f"Before Parameter {name} is on device: {param.device}")

        # Explicitly set all model parameters to the primary device
        #self.density_mlp = self.density_mlp.to(cfg.secondary_gpus[0])

        # Wrap with DataParallel
        self.density_mlp = \
            nn.DataParallel(
            self.density_mlp,
            device_ids=cfg.secondary_gpus,
            output_device=cfg.primary_gpus[0]
        )
        for name, param in self.density_mlp.named_parameters():
            print(f"After Parameter {name} is on device: {param.device}")

        # pose decoder MLP
        self.pose_decoder = \
            load_pose_decoder(cfg.pose_decoder.module)(
                embedding_size=cfg.pose_decoder.embedding_size,
                mlp_width=cfg.pose_decoder.mlp_width,
                mlp_depth=cfg.pose_decoder.mlp_depth)
    

    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self

    def print_parameter_devices(self, model):
        for name, param in model.named_parameters():
            print(f"Parameter {name} is on device: {param.device}")

    def compute_nearest_vertex_idx(self,sampled_points_flat, smpl_vertices, chunk_size):
        nearest_indices = []
        for i in range(0, sampled_points_flat.shape[0], chunk_size):
            chunked_points = sampled_points_flat[i:i + chunk_size]
            with torch.no_grad():
                distances = torch.cdist(chunked_points, smpl_vertices)   # [N_rays * N_samples, 6890]
                nearest_vertex_idx = torch.argmin(distances, dim=1)
                nearest_indices.append(nearest_vertex_idx)
        return torch.cat(nearest_indices)

    def interpolate_density(self, tri_feats, sampled_points, smpl_vertices):

        N_rays, N_samples, _ = sampled_points.shape
        self.print_parameter_devices(self.density_mlp)
        print(f"Primary GPU: {cfg.primary_gpus}, Secondary GPUs: {cfg.secondary_gpus}")

        # Find the nearest vertex for each sampled point
        sampled_points_flat = sampled_points.reshape(-1, 3)  # [N_rays * N_samples, 3]
        chunk = cfg.netchunk_per_gpu * len(cfg.secondary_gpus)
        nearest_vertex_idx = self.compute_nearest_vertex_idx(sampled_points_flat, smpl_vertices,chunk)
        if check_for_nans("nearest_vertex_idx", nearest_vertex_idx):
            print("NaN detected in nearest_vertex_idx")
        print(f"nearest_vertex_idx {nearest_vertex_idx.shape}")
        # Fetch the TriPlane features of the nearest vertices
        nearest_tri_feats = tri_feats[nearest_vertex_idx]  # [N_rays * N_samples, feature_dim]
        if check_for_nans("nearest_tri_feats", nearest_tri_feats):
            print("NaN detected in nearest_tri_feats")
        print(f"nearest_tri_feats {nearest_tri_feats.shape}")
        print(f"TriPlane features are on device: {tri_feats.device}")

        # nearest_tri_feats = nearest_tri_feats.to(next(self.density_mlp.parameters()).device)
        print(f"Nearest TriPlane features are on device before: {nearest_tri_feats.device}")
        # Predict density using the density MLP
        scaler = torch.cuda.amp.GradScaler()
        self.density_mlp.to(cfg.secondary_gpus[0])
        nearest_tri_feats = nearest_tri_feats.to(cfg.secondary_gpus[0])
        print(f"Density MLP is on device: {next(self.density_mlp.parameters()).device}")
        print(f"Nearest TriPlane features are on device: {nearest_tri_feats.device}")
        with torch.cuda.amp.autocast():
            interpolated_densities = self.density_mlp(nearest_tri_feats).to(cfg.secondary_gpus[0])# [N_rays * N_samples, 1]
        print(f"interpolated_densities {interpolated_densities.shape}")
        if check_for_nans("interpolated_densities", interpolated_densities):
            print("NaN detected in interpolated_densities")
        # Reshape back to [N_rays, N_samples, 1]
        return interpolated_densities.reshape(N_rays, N_samples)

    def blend_densities(self, nerf_density, triplane_density, weight=0.5):
        common_device = nerf_density.device
        triplane_density = triplane_density.to(common_device)
        if check_for_nans("nerf_density", nerf_density):
            print("NaN detected in nerf_density")
        if check_for_nans("triplane_density", triplane_density):
            print("NaN detected in triplane_density")

        print(f"In Blend Density Function {type(weight)} {weight}")
        print(f"In Blend Density Function {type(nerf_density)} {nerf_density.shape} {nerf_density.device}")
        print(f"In Blend Density Function {type(triplane_density)} {triplane_density.shape} {triplane_density.device}")
        blend_weight = torch.sigmoid(self.blending_weight)
        return (blend_weight * nerf_density) + ((1 - blend_weight) * triplane_density)

    def _query_mlp(
            self,
            pos_xyz,
            pos_embed_fn, 
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input,
            tri_feats,
            **kwargs):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)

        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        pos_embed_fn=pos_embed_fn,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        chunk=chunk,
                        tri_feats=tri_feats)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            pos_embed_fn,
            non_rigid_mlp_input,
            non_rigid_pos_embed_fn,
            chunk,
            tri_feats):
        raws = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end]


            if not cfg.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz)
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem)
                )
                xyz = result['xyz']

            xyz_embedded = pos_embed_fn(xyz)
            print(xyz_embedded.shape)
            raws += [self.cnl_mlp(
                        pos_embed=xyz_embedded)]
            # volume = self.triplane_mlp(tri_feats)

        output = {}
        output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0])

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            print(f"ret: {ret}")
            for k in ret:
                print(k)
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]
        print(f"rgb and alpha : {rgb.shape} {alpha.shape}")
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
        print(f"acc rgb and acc_map and depth_map: {rgb_map.shape} {acc_map.shape} {depth_map.shape}")
        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.

        return rgb_map, acc_map, weights, depth_map


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0 
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]

        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            **kwargs):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)

        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        tri_feats = kwargs['tri_feats']  # TriPlane features for SMPL vertices [6890, feature_dim]
        print(f"tri_feats shape: {tri_feats.shape}")
        smpl_vertices = kwargs['smpl_vertices']
        print(f"smpl shape: {smpl_vertices.shape}")
        triplane_density = self.interpolate_density(tri_feats, pts, smpl_vertices)  # [N_rays, N_samples]
        print(f"triplane_density shape after return: {triplane_density.shape}")
        mv_output = self._sample_motion_fields(
                            pts=pts,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'])
        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']

        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                                **kwargs)
        raw = query_result['raws']
        nerf_density = raw[..., 3]  # Extract NeRF densities [N_rays, N_samples]
        print(f"raw in render rays: {raw.shape}")
        print(f"triplane_density type in _render_rays: {type(triplane_density)} {triplane_density.shape}")
        triplane_density = triplane_density.squeeze(-1)
        print(f"triplane_density type in _render_rays: {type(triplane_density)} {triplane_density.shape}")
        print(f"nerf_density type in _render_rays: {type(nerf_density)} {nerf_density.shape}")
        # Blend densities from NeRF and TriPlane
        blended_density = self.blend_densities(nerf_density, triplane_density.squeeze(-1))  # [N_rays, N_samples]
        if check_for_nans("blended_density", blended_density):
            print("NaN detected in blended_density")
        # Replace NeRF density with blended density in the raw output
        raw[..., 3] = blended_density
        print(f"blended_density")
        rgb_map, acc_map, _, depth_map = \
            self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)

        return {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map}


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                motion_weights_priors,
                dst_posevec=None,
                near=None, far=None,
                betas=None,
                iter_val=1e7,
                **kwargs):

        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]
        print(f"Test:")
        print(f"iter_val:{iter_val}")
        print(f"cfg.pose_decoder.get('kick_in_iter', 0): {cfg.pose_decoder.get('kick_in_iter', 0)}")
        # correct body pose
        if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
            pose_out = self.pose_decoder(dst_posevec)
            # print(f"POSE_OUT: {dst_posevec}")
            # print(f"POSE_OUT: {pose_out}")

            refined_Rs = pose_out['Rs']
            refined_Ts = pose_out.get('Ts', None)
            print(f"refined_Rs shape: {refined_Rs.shape}")
            # print(f"refined_Rs: {refined_Rs}")
            if check_for_nans("refined_Rs", refined_Rs):
                print("NaN detected in refined_Rs")
            if refined_Ts is not None and check_for_nans("refined_Ts", refined_Ts):
                print("NaN detected in refined_Ts")

            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(
                                        dst_Rs_no_root, 
                                        refined_Rs)
            print(f"dst_Rs_no_root: {dst_Rs_no_root.shape}")
            if check_for_nans("dst_Rs_no_root", dst_Rs_no_root):
                print("NaN detected in dst_Rs_no_root")

            # print(f"dst_Rs_no_root: {dst_Rs_no_root}")
            # print(f"dst_Rs: {dst_Rs[:,0, ...]}")
            dst_Rs = torch.cat(
                [dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

            print(f"dst_Rs after cat: {dst_Rs.shape}")
            #convert matrix to vector
            import cv2, numpy as np

            # Apply cv2.Rodrigues to each (3, 3) matrix, then reshape to [1, 24, 3]
            dst_Rs_vec = np.array([cv2.Rodrigues(dst_Rs[0, i].detach().cpu().numpy())[0] for i in range(24)]).reshape(1,24,3).squeeze().flatten()
            print(dst_Rs_vec.shape)
            sex = 'neutral'
            smpl_model = SMPL(sex, model_dir=MODEL_DIR)
            if betas is not None:
                print("betas")
                vertices, joints = smpl_model(np.zeros(72), betas.cpu().numpy())
                # Normalize vertices for triplane input using NumPy functions
                vertices_norm = (vertices - vertices.min(axis=0, keepdims=True)) / \
                                (vertices.max(axis=0, keepdims=True) - vertices.min(axis=0, keepdims=True))
                if np.isnan(vertices_norm).any():
                    print("NaN detected in vertices_norm")
                # Extract TriPlane features on the SMPL vertices
                tri_feats = self.triplane(vertices_norm)
                if check_for_nans("tri_feats", tri_feats):
                    print("NaN detected in tri_feats")

                # Convert vertices to PyTorch tensor and move to GPU (same device as triplane)
                device = next(self.triplane.parameters()).device
                smpl_vertices = torch.tensor(vertices_norm, device=device).float()  # [6890, 3]
                kwargs.update({"tri_feats": tri_feats,
                               "smpl_vertices": smpl_vertices})

                print(f"tri_feats: {tri_feats.shape}")


            if refined_Ts is not None:
                dst_Ts = dst_Ts + refined_Ts

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,                         
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)

        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input
        })

        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts,
                                            cnl_gtfms=cnl_gtfms)
        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension
        if check_for_nans("motion_weights_vol", motion_weights_vol):
            print("NaN detected in motion_weights_vol")
        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': motion_weights_vol
        })

        rays_o, rays_d = rays
        rays_shape = rays_d.shape 

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        for k in all_ret:
            if check_for_nans(f"all_ret[{k}]", all_ret[k]):
                print(f"NaN detected in all_ret[{k}]")

            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret
