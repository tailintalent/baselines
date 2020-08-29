"""
An interface for asynchronous vectorized environments.
"""

import multiprocessing as mp
import numpy as np
from .vec_env import VecEnv, CloudpickleWrapper, clear_mpi_env_vars
import ctypes
from baselines import logger
import torch

from .util import dict_to_obs, obs_space_info, obs_to_dict

_NP_TO_CT = {np.float32: ctypes.c_float,
             np.int32: ctypes.c_int32,
             np.int8: ctypes.c_int8,
             np.uint8: ctypes.c_char,
             np.bool: ctypes.c_bool}


class ShmemVecEnv(VecEnv):
    """
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    """

    def __init__(self, env_fns, spaces=None, context='spawn', is_shared_memory=True, is_numpy=False, is_torch=False):
        """
        If you don't specify observation_space, we'll have to create a dummy
        environment to get it.
        """
        ctx = mp.get_context(context)
        self.is_shared_memory = is_shared_memory
        self.is_numpy = is_numpy
        self.is_torch = is_torch
        if spaces:
            observation_space, action_space = spaces
        else:
            logger.log('Creating dummy env object to get spaces')
            with logger.scoped_configure(format_strs=[]):
                dummy = env_fns[0]()
                observation_space, action_space = dummy.observation_space, dummy.action_space  # e.g. Box(1, 84, 84), discrete(6)
                dummy.close()
                del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)  # ([None], {None: (1, 84, 84)}, {None: dtype('uint8')})
        if self.is_shared_memory:
            self.obs_bufs = [
                {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}  # {None: ctx.Array(<class 'ctypes.c_char'>, 1 * 84 * 84)}
                for _ in env_fns]
        else:
            self.obs_bufs = [None for _ in env_fns]
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
                # obs_buf: {None: <SynchronizedString wrapper for <multiprocessing.sharedctypes.c_char_Array_7056 object at 0x7f09f74decb0>>}
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(target=_subproc_worker,
                            args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys, self.is_shared_memory))
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            logger.warn('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        return self._decode_obses([pipe.recv() for pipe in self.parent_pipes])  # np.array with [16, 1, 84, 84]

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))
        self.waiting_step = True

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        self.waiting_step = False
        obs, rews, dones, infos = zip(*outs)
        rews = np.array(rews)
        if self.is_torch:
            rews = torch.from_numpy(rews)[:, None].float()
        return self._decode_obses(obs), rews, np.array(dones), infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self, mode='human'):
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_obses(self, obs):
        if self.is_shared_memory:
            result = {}
            for k in self.obs_keys:
                bufs = [b[k] for b in self.obs_bufs]  # b[k]: <SynchronizedString wrapper for <multiprocessing.sharedctypes.c_char_Array_7056 object at 0x7ff9808a0a70>>
                o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) for b in bufs]
                result[k] = np.array(o)
            return dict_to_obs(result)  # [16, 1, 84, 84]
        else:
            if self.is_numpy:
                return np.array(obs)
            else:
                return list(obs)


def _subproc_worker(pipe, parent_pipe, env_fn_wrapper, obs_bufs, obs_shapes, obs_dtypes, keys, is_shared_memory=True):
    """
    Control a single environment instance using IPC and
    shared memory.
    """
    def _write_obs(maybe_dict_obs):
        if is_shared_memory:
            flatdict = obs_to_dict(maybe_dict_obs)  # maybe_dict_obs: [1, 84, 84], flatdict: {None: [1, 84, 84]}
            for k in keys:
                if obs_bufs[k] is not None:
                    dst = obs_bufs[k].get_obj()  # <multiprocessing.sharedctypes.c_char_Array_7056 object at 0x7fe1a78833b0>
                    dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])  # pylint: disable=W0212    # dst_np: [1, 84, 84], obs_dtypes: {None: dtype('uint8')}
                    np.copyto(dst_np, flatdict[k])
        else:
            return maybe_dict_obs

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                pipe.send(_write_obs(env.reset()))
            elif cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                pipe.send((_write_obs(obs), reward, done, info))
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError('Got unrecognized cmd %s' % cmd)
    except KeyboardInterrupt:
        print('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()
