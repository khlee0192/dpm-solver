import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import inverse_data_transform
import torchvision.transforms as transforms
from PIL import Image

from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


# global variables for checking
x_T_first = []
x_0_second = []
x_T_third = []
x_0_fourth = []
# gvar_bwd = []

# gvar_gen = {"t":[], "x":[]}
# gvar_det = {"t":[], "x":[]}

class Diffusion_and_Inversion(Diffusion): 
    # If we want to edit codes, we should use inheritance 
    # https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)
    def sample_image(self, x, model, last=True, classifier=None, base_samples=None):
        # We re-define this function to:
        # 1. save intermediate results.
        # 2. perform inversion after diffusion
        # input is x_T (noise)

        # for using global variable (for saving intermediate results)
        global x_T_first, x_0_second, x_T_third, x_0_fourth
        x_T_first.append(x.clone()) 

        # original code of sample_image
        assert last
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        classifier_scale = self.config.sampling.classifier_scale if self.args.scale is None else self.args.scale
        if self.config.sampling.cond_class:
            if self.args.fixed_class is None:
                classes = torch.randint(low=0, high=self.config.data.num_classes, size=(x.shape[0],)).to(x.device)
            else:
                classes = torch.randint(low=self.args.fixed_class, high=self.args.fixed_class + 1, size=(x.shape[0],)).to(x.device)
        else:
            classes = None
        
        if base_samples is None:
            if classes is None:
                model_kwargs = {}
            else:
                model_kwargs = {"y": classes}
        else:
            model_kwargs = {"y": base_samples["y"], "low_res": base_samples["low_res"]}

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = generalized_steps(x, seq, model_fn, self.betas, eta=self.args.eta, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        return torch.split(out, 3, dim=1)[0]
                return out
            xs, _ = ddpm_steps(x, seq, model_fn, self.betas, classifier=classifier, is_cond_classifier=self.config.sampling.cond_class, classifier_scale=classifier_scale, **model_kwargs)
            x = xs[-1]
        elif self.args.sample_type in ["dpmsolver", "dpmsolver++"]:
            from dpm_solver.sampler import NoiseScheduleVP, model_wrapper
            def model_fn(x, t, **model_kwargs):
                out = model(x, t, **model_kwargs)
                # If the model outputs both 'mean' and 'variance' (such as improved-DDPM and guided-diffusion),
                # We only use the 'mean' output for DPM-Solver, because DPM-Solver is based on diffusion ODEs.
                if "out_channels" in self.config.model.__dict__.keys():
                    if self.config.model.out_channels == 6:
                        out = torch.split(out, 3, dim=1)[0]
                return out

            def classifier_fn(x, t, y, **classifier_kwargs):
                logits = classifier(x, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return log_probs[range(len(logits)), y.view(-1)]

            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
            model_fn_continuous = model_wrapper(
                model_fn,
                noise_schedule,
                model_type="noise",
                model_kwargs=model_kwargs,
                guidance_type="uncond" if classifier is None else "classifier",
                condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
                guidance_scale=classifier_scale,
                classifier_fn=classifier_fn,
                classifier_kwargs={},
            )
            dpm_solver = DPM_Solver_inv(
                model_fn_continuous,
                noise_schedule,
                algorithm_type=self.args.sample_type,
                correcting_x0_fn="dynamic_thresholding" if self.args.thresholding else None,
            )
            x, intermediate_gen = dpm_solver.sample(
                x,
                steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                order=self.args.dpm_solver_order,
                skip_type=self.args.skip_type,
                method=self.args.dpm_solver_method,
                lower_order_final=self.args.lower_order_final,
                denoise_to_zero=self.args.denoise,
                solver_type=self.args.dpm_solver_type,
                atol=self.args.dpm_solver_atol,
                rtol=self.args.dpm_solver_rtol,
                return_intermediate=True,
            )
            #gvar_gen["x"] = intermediate_gen
            #gvar_gen["x"].insert(0, gvar_fwd)
            x_0_second.append(x.clone())
            # TODO: x = quantize(x), but I haven't implented yet.

            # DPM-solver inversion. This function is also re-defined in this .py file.
            y, intermediate_det = dpm_solver.inversion(
                x,
                steps=self.args.inv_timesteps,
                order=self.args.inv_order,
                skip_type=self.args.skip_type,
                method=self.args.dpm_solver_method,
                lower_order_final=self.args.lower_order_final,
                denoise_to_zero=self.args.denoise,
                solver_type=self.args.dpm_solver_type,
                atol=self.args.dpm_solver_atol,
                rtol=self.args.dpm_solver_rtol,
                inverse_opt=not self.args.inv_naive,
                return_intermediate=True,
            )

            # Save intermediate and the last results
            x_T_third.append(y.clone())
            # gvar_det["x"] = intermediate_det
            # gvar_det["x"].insert(0, x)
            # list_to_images(self.config, gvar_gen["x"], "./inv_results_gen")
            # list_to_images(self.config, gvar_det["x"], "./inv_results_det")

            x = dpm_solver.sample(
                y,
                steps=(self.args.timesteps - 1 if self.args.denoise else self.args.timesteps),
                order=self.args.dpm_solver_order,
                skip_type=self.args.skip_type,
                method=self.args.dpm_solver_method,
                lower_order_final=self.args.lower_order_final,
                denoise_to_zero=self.args.denoise,
                solver_type=self.args.dpm_solver_type,
                atol=self.args.dpm_solver_atol,
                rtol=self.args.dpm_solver_rtol,
                return_intermediate=False,
            )
            x_0_fourth.append(x.clone())
            i = len(x_T_first) - 1
            # print(i, "T0T ", (x_T_first[i]-x_T_third[i]).norm()/x_T_first[i].norm(),
            #       "0T0 ", (x_0_second[i]-x_0_fourth[i]).norm()/x_0_second[i].norm())
            torch.save(x_T_first, os.path.join(self.args.exp, 'first.pth'))
            torch.save(x_0_second, os.path.join(self.args.exp, 'second.pth'))
            torch.save(x_T_third, os.path.join(self.args.exp, 'third.pth'))
            torch.save(x_0_fourth, os.path.join(self.args.exp, 'fourth.pth'))
            
        else:
            raise NotImplementedError
        
        # concatenated_tensor = torch.cat(gvar_det['x'], dim=3)
        # image = transforms.ToPILImage()(concatenated_tensor[0])
        # image.show()
        
        return x, classes


def list_to_images(config, list, path):
    len_list = len(list)
    os.makedirs(path, exist_ok=True)
    for i in range(len_list):
        image = transforms.ToPILImage()(inverse_data_transform(config, list[i])[0])
        output_path = os.path.join(path, str(i) + "_" + str(len_list) + ".png")
        image.save(output_path)


from dpm_solver.sampler import DPM_Solver
class DPM_Solver_inv(DPM_Solver):
    # If we want to edit codes, we should use inheritance 
    # https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)
    # TODO: def sample (for checking, yank global var), but functionally not necessary
    def inversion(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='dpmsolver',
        atol=0.0078, rtol=0.05, return_intermediate=False, inverse_opt=False, new_method = True,
    ):
        # the same code
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        # if return_intermediate:
        #     assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        # if self.correcting_xt_fn is not None:
        #     assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []

        if not new_method or order==1:
            with torch.no_grad():
                if method == 'multistep' or method == 'singlestep':
                    # assert order == 1 # for now, only DDIM inversion is possible
                    timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                    assert timesteps.shape[0] - 1 == steps
                    # Init the initial values
                    # step = steps
                    # t = timesteps[step]
                    # t_prev_list = [t]
                    for step in range(steps, 1, -1):
                        t = timesteps[step] 
                        s = timesteps[step-1] # s is more closer to T than t
                        r = timesteps[step-2]
                        # x = self.multistep_dpm_solver_update 가 하는게 뭐지?
                        # order==1 일 경우엔 self.dpm_solver_first_update(x, s, t, model_s=model_prev_list[-1])
                        
                        # coefficients for dpmsolver++
                        ns = self.noise_schedule
                        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
                        h = lambda_t - lambda_s
                        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
                        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
                        alpha_t = torch.exp(log_alpha_t)

                        # dpmsolver++
                        assert self.algorithm_type == "dpmsolver++"
                        phi_1 = torch.expm1(-h)
                        model_s = self.model_fn(x, s)
                        x_t = x

                        # naive DDIM inversion
                        x = (
                            sigma_s / sigma_t * x
                            + sigma_s / sigma_t * alpha_t * phi_1 * model_s
                        )
                        if inverse_opt:
                            torch.set_grad_enabled(True)
                            x = self.differential_correction(x, s, t, x_t, order=order, r=r)
                            torch.set_grad_enabled(False)
                        if return_intermediate:
                            intermediates.append(x) 
                    for step in range(1, 0, -1):
                        t = timesteps[step] 
                        s = timesteps[step-1] # s is more closer to T than t
                        # x = self.multistep_dpm_solver_update 가 하는게 뭐지?
                        # order==1 일 경우엔 self.dpm_solver_first_update(x, s, t, model_s=model_prev_list[-1])
                        
                        # coefficients for dpmsolver++
                        ns = self.noise_schedule
                        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
                        h = lambda_t - lambda_s
                        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
                        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
                        alpha_t = torch.exp(log_alpha_t)

                        # dpmsolver++
                        assert self.algorithm_type == "dpmsolver++"
                        phi_1 = torch.expm1(-h)
                        model_s = self.model_fn(x, s)
                        x_t = x

                        # naive DDIM inversion
                        x = (
                            sigma_s / sigma_t * x
                            + sigma_s / sigma_t * alpha_t * phi_1 * model_s
                        )
                        if inverse_opt:
                            torch.set_grad_enabled(True)
                            x = self.differential_correction(x, s, t, x_t, order=1)
                            torch.set_grad_enabled(False)
                        if return_intermediate:
                            intermediates.append(x)     
        else: # 0731
            with torch.no_grad():
                if method == 'multistep' or method == 'singlestep':
                    # assert order == 1 # for now, only DDIM inversion is possible
                    timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                    assert timesteps.shape[0] - 1 == steps
                    # Init the initial values
                    # step = steps
                    # t = timesteps[step]
                    # t_prev_list = [t]
                    for outerstep in range(steps, 1, -1):
                        y = x.clone()
                        for step in range(outerstep, outerstep-2, -1):                    
                            t = timesteps[step] 
                            s = timesteps[step-1] # s is more closer to T than t
                            r = timesteps[step-2]
                            # x = self.multistep_dpm_solver_update 가 하는게 뭐지?
                            # order==1 일 경우엔 self.dpm_solver_first_update(x, s, t, model_s=model_prev_list[-1])
                            
                            # coefficients for dpmsolver++
                            ns = self.noise_schedule
                            lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
                            h = lambda_t - lambda_s
                            log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
                            sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
                            alpha_t = torch.exp(log_alpha_t)

                            # dpmsolver++
                            assert self.algorithm_type == "dpmsolver++"
                            phi_1 = torch.expm1(-h)
                            model_s = self.model_fn(x, s)
                            y_t = y

                            # naive DDIM inversion
                            y = (
                                sigma_s / sigma_t * y
                                + sigma_s / sigma_t * alpha_t * phi_1 * model_s
                            )
                            if inverse_opt:
                                torch.set_grad_enabled(True)
                                y = self.differential_correction(y, s, t, y_t, order=order, r=r)
                                torch.set_grad_enabled(False)
                        
                        # outer step
                        t = timesteps[outerstep] 
                        s = timesteps[outerstep-1] # s is more closer to T than t
                        r = timesteps[outerstep-2]
                        # x = self.multistep_dpm_solver_update 가 하는게 뭐지?
                        # order==1 일 경우엔 self.dpm_solver_first_update(x, s, t, model_s=model_prev_list[-1])
                        
                        # coefficients for dpmsolver++
                        ns = self.noise_schedule
                        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
                        h = lambda_t - lambda_s
                        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
                        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
                        alpha_t = torch.exp(log_alpha_t)

                        # dpmsolver++
                        assert self.algorithm_type == "dpmsolver++"
                        phi_1 = torch.expm1(-h)
                        model_s = self.model_fn(x, s)
                        x_t = x
                        if not inverse_opt:
                            # naive DDIM inversion
                            x = (
                                sigma_s / sigma_t * x
                                + sigma_s / sigma_t * alpha_t * phi_1 * model_s
                            )

                        if inverse_opt:
                            model_s_output = self.model_fn(y_t, s)
                            model_r_output = self.model_fn(y, r)
                            # not naive DDIM inversion
                            x = (
                                sigma_s / sigma_t * x
                                + sigma_s / sigma_t * alpha_t * phi_1 * model_s_output
                            )
                            torch.set_grad_enabled(True)
                            x = self.differential_correction(x, s, t, x_t, order=order, r=r, 
                                                             model_s_output = model_s_output, model_r_output = model_r_output)
                            torch.set_grad_enabled(False)            
                        if return_intermediate:
                            intermediates.append(x) 

                    for step in range(1, 0, -1):
                        t = timesteps[step] 
                        s = timesteps[step-1] # s is more closer to T than t
                        # x = self.multistep_dpm_solver_update 가 하는게 뭐지?
                        # order==1 일 경우엔 self.dpm_solver_first_update(x, s, t, model_s=model_prev_list[-1])
                        
                        # coefficients for dpmsolver++
                        ns = self.noise_schedule
                        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
                        h = lambda_t - lambda_s
                        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
                        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
                        alpha_t = torch.exp(log_alpha_t)

                        # dpmsolver++
                        assert self.algorithm_type == "dpmsolver++"
                        phi_1 = torch.expm1(-h)
                        model_s = self.model_fn(x, s)
                        x_t = x

                        # naive DDIM inversion
                        x = (
                            sigma_s / sigma_t * x
                            + sigma_s / sigma_t * alpha_t * phi_1 * model_s
                        )
                        if inverse_opt:
                            torch.set_grad_enabled(True)
                            x = self.differential_correction(x, s, t, x_t, order=1)
                            torch.set_grad_enabled(False)    
                        if return_intermediate:
                            intermediates.append(x)         
        if return_intermediate:
            return x, intermediates
        else:
            return x
    
    def differential_correction(self, x, s, t, x_t, r=None, order=1,
                                 use_float=False, n_iter=100, lr=0.05, th=1e-6,
                                 model_s_output=None, model_r_output=None):
        # order=1
        # order=2, multistep
        if order==1:
            import copy
            model_fn_copy = copy.deepcopy(self.model_fn)
            input = x.clone()
            x_t = x_t.clone()
            if use_float:
                model_fn_copy = model_fn_copy.float()
                input = input.float()
                x_t = x_t.float() 
            input.requires_grad_(True)
            loss_function = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.SGD([input], lr=lr)
            for i in range(n_iter):
                model_output = model_fn_copy(input, s) # estimated noise
                x_t_pred = self.dpm_solver_first_update(input, s, t, model_s=model_output)
                loss = loss_function(x_t_pred, x_t)
                #print(f"t: {t}, Iteration {i}, Loss: {loss.item():.6f}")
                if loss.item() < th:
                    break             
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return input            
        elif order==2:
            assert r is not None
            # only for multistep
            import copy
            model_fn_copy = copy.deepcopy(self.model_fn)
            input = x.clone()

            x_t = x_t.clone()   
            if use_float:
                model_fn_copy = model_fn_copy.float()
                input = input.float()
                x_t = x_t.float() 
            input.requires_grad_(True)
            loss_function = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.SGD([input], lr=lr)
            
            # for 2nd order correction
            ns = self.noise_schedule
            model_t_output = model_fn_copy(x_t, t).detach()
            lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(r), ns.marginal_lambda(s), ns.marginal_lambda(t)
            log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
            sigma_prev_0, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
            alpha_t = torch.exp(log_alpha_t)
            h_0 = lambda_prev_0 - lambda_prev_1
            h = lambda_t - lambda_prev_0
            r0 = h_0 / h
            phi_1 = torch.expm1(-h)

            for i in range(n_iter):
                model_output = model_fn_copy(input, s) # estimated noise
                x_t_pred = self.dpm_solver_first_update(input, s, t, model_s=model_output)
                # 2nd order correction
                # diff = (1. / r0) * (model_t_output - model_output)
                if model_s_output is not None and model_r_output is not None:
                    diff =  (1./ r0) * (model_s_output - model_r_output)
                else:
                    diff = 1. * (model_t_output - model_output)
                x_t_pred = x_t_pred - 0.5 * alpha_t * phi_1 * diff

                loss = loss_function(x_t_pred, x_t)
                #print(f"t: {t:.3f}, Iteration {i}, Loss: {loss.item():.6f}")
                if loss.item() < th:
                    break             
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return input
        else:
            raise NotImplementedError

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--inversion", type=bool, default=False, help="DDIM inversion"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=False,
        default="",
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument("--test", action="store_true", help="Whether to test the model")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach ('generalized'(DDIM) or 'ddpm_noisy'(DDPM) or 'dpmsolver' or 'dpmsolver++')",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="time_uniform",
        help="skip according to ('uniform' or 'quadratic' for DDIM/DDPM; 'logSNR' or 'time_uniform' or 'time_quadratic' for DPM-Solver)",
    )
    parser.add_argument(
        "--base_samples",
        type=str,
        default=None,
        help="base samples for upsampling, *.npz",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--dpm_solver_order", type=int, default=3, help="order of dpm-solver"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--fixed_class", type=int, default=None, help="fixed class label for conditional sampling"
    )
    parser.add_argument(
        "--dpm_solver_atol", type=float, default=0.0078, help="atol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_rtol", type=float, default=0.05, help="rtol for adaptive step size algorithm"
    )
    parser.add_argument(
        "--dpm_solver_method",
        type=str,
        default="singlestep",
        help="method of dpm_solver ('adaptive' or 'singlestep' or 'multistep' or 'singlestep_fixed'",
    )
    parser.add_argument(
        "--dpm_solver_type",
        type=str,
        default="dpm_solver",
        help="type of dpm_solver ('dpm_solver' or 'taylor'",
    )
    parser.add_argument(
        "--inv_naive", action='store_true', default=False, help="Naive DDIM of inversion"
    )
    parser.add_argument(
        "--inv_timesteps", type=int, default=1000, help="number of steps of inversion"
    )
    parser.add_argument(
        "--inv_order", type=int, default=2, help="order of inversion"
    )
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--lower_order_final", action="store_true", default=False)
    parser.add_argument("--thresholding", action="store_true", default=False)
    
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument("--port", type=str, default="12355")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    #with open(os.path.join("configs", args.config), "r") as f:
    with open(os.path.join("./examples/ddpm_and_guided-diffusion/configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)

    if not args.test and not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError("level {} not supported".format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)
            args.image_folder = os.path.join(
                args.exp, "image_samples", args.image_folder
            )
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                if not (args.fid or args.interpolation):
                    overwrite = False
                    if args.ni:
                        overwrite = True
                    else:
                        response = input(
                            f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                        )
                        if response.upper() == "Y":
                            overwrite = True

                    if overwrite:
                        shutil.rmtree(args.image_folder)
                        os.makedirs(args.image_folder)
                    else:
                        print("Output image folder exists. Program halted.")
                        sys.exit(0)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def evaluate(args):
    # torch.save(x_T_first, os.path.join(self.args.exp, 'first.pth'))
    # torch.save(x_0_second, os.path.join(self.args.exp, 'second.pth'))
    # torch.save(x_T_third, os.path.join(self.args.exp, 'thrid.pth'))
    # torch.save(x_0_fourth, os.path.join(self.args.exp, 'fourth.pth'))    
    file1_path = os.path.join(args.exp, 'first.pth')
    file2_path = os.path.join(args.exp, 'second.pth')
    file3_path = os.path.join(args.exp, 'third.pth')
    file4_path = os.path.join(args.exp, 'fourth.pth')
    t1 = torch.load(file1_path)
    t2 = torch.load(file2_path)
    t3 = torch.load(file3_path)
    t4 = torch.load(file4_path)
    recon_err_T0T = []
    recon_err_0T0 = [] 
    for i in range(len(t1)):
        recon_err_T0T.append( (((t1[i]-t3[i]).norm()/(t1[i].norm())).item())**2 )
        recon_err_0T0.append( (((t2[i]-t4[i]).norm()/(t2[i].norm())).item())**2 )
    import statistics
    data = recon_err_T0T
    mean_value = statistics.mean(data)
    std_value = statistics.stdev(data)

    # 결과 출력
    print("T0T")
    print("평균(mean):", mean_value)
    print("표준 편차(std):", std_value)
    data=recon_err_0T0
    mean_value = statistics.mean(data)
    std_value = statistics.stdev(data)

    # 결과 출력
    print("0T0")
    print("평균(mean):", mean_value)
    print("표준 편차(std):", std_value)

def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    print(args.inv_naive)
    world_size = torch.cuda.device_count()
    mp.spawn(sample,
            args=(world_size, args, config),
            nprocs=world_size,
            join=True)
    
    evaluate(args)


def sample(rank, world_size, args, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
        # set random seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    try:
        runner = Diffusion_and_Inversion(args, config, rank=rank)
        if args.sample:
            runner.sample()
        elif args.test:
            runner.test()
        else:
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())
    dist.destroy_process_group()

if __name__ == "__main__":
    sys.exit(main())