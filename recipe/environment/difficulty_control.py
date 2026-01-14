import math, random
import json
from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Tuple, Dict, Optional
import numpy as np
@dataclass
class BetaScheduler:
    beta: float = 2
    beta_min: float = 0.5
    beta_max: float = 5
    eps: float = 1e-12
    m: int = 5                                   # 历史窗口长度
    variant: Literal['bb1','bb2','alt'] = 'bb1'  # 'alt' 表示在 bb1/bb2 间交替

    # 内部状态
    prev_distance: Optional[float] = None
    prev_err: Optional[float] = None
    _hist: Deque[Tuple[float, float]] = None     # 存 (s, y)
    _alt_flag: int = 0

    def __post_init__(self):
        if self._hist is None:
            self._hist = deque(maxlen=self.m)

    def update_beta(self,err,distance) -> float:
        """
        需要上一轮的 accuracy 与 distance（或你自己在外部保存），
        这样本轮就能形成 (s, y) 并入历史窗口。
        """
        if self.prev_distance != None and self.prev_err != None:

            s = float(distance) - float(self.prev_distance)
            y = err - self.prev_err
            
            self._push_pair(s, y)

        bb = self._estimate_bb()
        if bb is None:
            new_beta = self.beta  # 没法估计时保持原值（也可改为取 beta_min）
        else:
            new_beta = max(self.beta_min, min(bb, self.beta_max))

        self.beta = new_beta
        self.prev_err = err
        self.prev_distance = distance
        return new_beta

    def _push_pair(self, s: float, y: float):
        # 仅在 |y| 足够大、且存在参数移动时记录，避免 0/0
        if abs(y) > self.eps and abs(s) > self.eps:
            self._hist.append((s, y))

    def _estimate_bb(self):
        if not self._hist:
            return None

        if self.variant == 'alt':
            use_bb1 = (self._alt_flag % 2 == 0)
            self._alt_flag += 1
        else:
            use_bb1 = (self.variant == 'bb1')
        if len(self._hist) == self.m:
            # 聚合：sum over window
            sum_ss = sum(s*s for s, _ in self._hist)
            sum_sy = sum(s*y for s, y in self._hist)
            sum_yy = sum(y*y for _, y in self._hist)

            if use_bb1:
                # α = sum s^T s / sum s^T y
                if abs(sum_sy) <= self.eps:
                    return None
                return abs(sum_ss / sum_sy)
            else:
                # α = sum s^T y / sum y^T y
                if abs(sum_yy) <= self.eps:
                    return None
                return abs(sum_sy / sum_yy)
        else:
            return None
        
    # ---- JSON 序列化（类内）----
    _SERDE_TAG = "BetaScheduler-v1"

    def _to_serializable(self) -> dict:
        """
        转为 JSON-safe 字典。
        - _hist(deque) 转成 [[s,y], ...]
        - 保留运行态 beta 与 _alt_flag
        """
        return {
            "type": self._SERDE_TAG,
            "params": {
                "beta": self.beta,
                "beta_min": self.beta_min,
                "beta_max": self.beta_max,
                "eps": self.eps,
                "m": self.m,
                "variant": self.variant,
            },
            "state": {
                "hist": [[float(s), float(y)] for (s, y) in (list(self._hist) if self._hist else [])],
                "alt_flag": int(self._alt_flag),
                "prev_distance":self.prev_distance,
                "prev_err":self.prev_err
            },
        }

    @classmethod
    def _from_serializable(cls, payload: dict) -> "BetaScheduler":
        """
        从 JSON-safe 字典恢复实例；按 m 恢复 deque(maxlen=m) 并回填历史。
        """
        if payload.get("type") != cls._SERDE_TAG:
            raise ValueError("Unrecognized BetaScheduler payload type")

        params = payload["params"]
        state = payload.get("state", {})

        obj = cls(
            beta=params["beta"],
            beta_min=params["beta_min"],
            beta_max=params["beta_max"],
            eps=params["eps"],
            m=params["m"],
            variant=params["variant"],
        )

        # 恢复内部状态
        obj._hist = deque(maxlen=obj.m)
        for pair in state.get("hist", []):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                s, y = float(pair[0]), float(pair[1])
                # 与 _push_pair 约束一致（也可直接 append 原样数据）
                if abs(y) > obj.eps and abs(s) > obj.eps:
                    obj._hist.append((s, y))

        obj._alt_flag = int(state.get("alt_flag", 0))
        obj.prev_err = state.get("prev_err",None)
        obj.prev_distance = state.get("prev_distance",None)
        return obj


import math
import random

class DifficultyControl:
    def __init__(self, 
                 target=0.5,
                 k=2,
                 dmin=0.0, dmax=50.0, 
                 step_cap=1, 
                 jitter=0.15,
                 ema_beta = 0.8,           # 目前没用，但保留参数不破坏旧接口
                 state = None,
                 activate_function="linear",
                 k_updater_name="fixed",
                 # ===== 新增：用于环境筛选的参数 =====
                 history_len=50,           # 记录最近多少个 d
                 slope_scale=0.05,         # 斜率尺度，~slope_scale 时 w_effective 就很高
                 age_scale=10_000,         # 未访问步数达到 age_scale 时 w_recency=1
                 std_scale=0.8,
                 alpha=1.0,                # 有效性权重系数
                 beta=0.5                  # 未访问时间权重系数
                 ):
        self.target = target
        self.k = k
        self.dmin, self.dmax = int(dmin), int(dmax)
        self.step_cap = step_cap
        self.jitter = jitter
        self.ema_beta = ema_beta   # 保留字段
        self.activate_function = activate_function

        self.history_len = history_len
        self.slope_scale = slope_scale
        self.age_scale = age_scale
        self.std_scale = std_scale
        self.alpha = alpha
        self.beta = beta

        if state:
            # 兼容旧的 state，如果没有 history 字段，就补一个空列表
            self.state = state
            if "distance_history" not in self.state:
                self.state["ditsance_history"] = []
            if "correct_history" not in self.state:
                self.state["correct_history"] = []
        else:
            self.state = {
                "d": 0.0,         # 当前连续难度
                "t": 0,           # 内部计数器
                "last_step": -1,  # 上次访问的全局 step，-1 表示从未访问
                "distance_history": [],     # 最近若干次 d 的历史
                "correct_history": []
            } 
        
        self.k_updater_name = k_updater_name
        if k_updater_name != "fixed":
            self.k_updater = BetaScheduler(beta=self.k)

    # ================= 基础功能 =================

    def propose_distances(self, batch_size):
        """
        按照当前连续难度 self.state['d']，拆成 floor/ceil 整数 difficulty。
        """
        st = self.state
        d = st["d"]
        lo, hi = math.floor(d), math.ceil(d)
        frac = d - lo
        num_hi = round(batch_size * frac)
        distances = [hi] * num_hi + [lo] * (batch_size - num_hi)
        random.shuffle(distances)
        return distances

    def update(self, distance_correct_avg_len_dict, now_step=None):
        """
        distance_correct_avg_len_dict:
            {distance: (correct_avg, leng), ...}
        now_step: 当前全局 step（用于 last_step 和之后的 recency 计算）
        """
        assert len(distance_correct_avg_len_dict) <= 2, \
            f"len distance_correct_avg_len_dict= {len(distance_correct_avg_len_dict)}  >2 "
        
        st = self.state
        distance = st['d']     # 当前控制的连续难度
        eps = 1e-7
        batch_avg_correct = 0.0
        batch_len = 0

        # # ====== 新增：记录当前这次使用的 difficulty 历史，用于后续斜率计算 ======
        # self._append_history(distance)

        batch_alpha = 0.0
        batch_ema = 0.0  # 这里会直接等价为加权的 batch_avg_correct

        for test_distance, (correct_avg, leng) in distance_correct_avg_len_dict.items():
            # 现在不再做 per-distance EMA，直接用 correct_avg
            if len(distance_correct_avg_len_dict) == 2:
                # 按照你原来的 alpha 规则做双点插值
                alpha = (1 - abs(distance - test_distance))
                batch_alpha += alpha
                batch_ema += alpha * correct_avg
            else:
                batch_ema = correct_avg
            
            batch_avg_correct += correct_avg * leng
            batch_len += leng
        
        if batch_len > 0:
            batch_avg_correct = batch_avg_correct / batch_len
        else:
            batch_avg_correct = 0.0

        if len(distance_correct_avg_len_dict) == 2:
            assert (abs(batch_alpha) <= eps or abs(batch_alpha - 1) <= eps), \
                f"batch_alpha {batch_alpha} not pass the alpha rule."
        else:
            batch_alpha = 1.0

        self._append_history(distance,batch_avg_correct)

        if self.activate_function.startswith("base"): 
            err = batch_avg_correct - self.target 
        else: 
            err = batch_ema - self.target

        # ----------- k update & delta compute -----------
        current_k = self.k
        if self.k_updater_name != "fixed":
            current_k = self.k_updater.update_beta(err=err, distance=distance) 
        
        raw_delta = current_k * err
        
        if self.activate_function in ("linear", "base"):
            delta = raw_delta
            if delta > self.step_cap:
                delta = self.step_cap
            if delta < -self.step_cap:
                delta = -self.step_cap
        elif self.activate_function in ("tanh", "base_tanh"):
            delta = math.tanh(3 * raw_delta) / math.tanh(3) * self.step_cap
        else:
            raise NotImplementedError
        
        if delta > self.step_cap:
            delta = self.step_cap
        if delta < -self.step_cap:
            delta = -self.step_cap

        st["d"] = max(self.dmin, min(self.dmax, st["d"] + delta))
        st["t"] += 1
        st['last_step'] = now_step if now_step is not None else st.get('last_step', -1)
        
        # 为了兼容旧接口，返回 [平滑正确率, 原始正确率, 当前 k]
        return [batch_ema, batch_avg_correct, current_k]

    # ================== 环境筛选相关：权重计算 ==================

    def _append_history(self, distance,correct):
        """往 state['history'] 里塞一条新的 d，并限制长度。"""
        hist = self.state.get("distance_history")
        if hist is None:
            hist = []
            self.state["distance_history"] = hist
        hist.append(float(distance))
        if len(hist) > self.history_len:
            # 保留最近 history_len 条
            self.state["distance_history"] = hist[-self.history_len:]
            
        
        hist = self.state.get("correct_history")
        if hist is None:
            hist = []
            self.state["correct_history"] = hist
        hist.append(float(correct))
        if len(hist) > self.history_len:
            # 保留最近 history_len 条
            self.state["correct_history"] = hist[-self.history_len:]
    def _compute_effective_weight(self):
        """
        根据最近 history_len 个 d 的线性回归斜率，计算 w_effective：
        - slope > 0 较大 → 说明难度整体还在涨 → w_effective 接近 1
        - slope ≈ 0 或负 → 难度整体不涨（只是抖动） → w_effective 接近 0
        """
        hist = self.state.get("distance_history", [])
        L = len(hist)
        if L < 3:
            # 数据太少，默认认为有训练价值
            return 1.0

        d_vals = hist
        # t = 0, 1, ..., L-1
        t_mean = (L - 1) / 2.0
        d_mean = sum(d_vals) / float(L)

        num = 0.0
        den = 0.0
        for i, d in enumerate(d_vals):
            dt = i - t_mean
            num += dt * (d - d_mean)
            den += dt * dt

        if den <= 1e-8:
            return 1.0

        slope = num / den
        slope_pos = max(0.0, slope)  # 只奖励“向上”的趋势

        # 缩放 + 平滑映射：slope≈slope_scale 时，w_effective 已经比较高
        s = slope_pos / float(self.slope_scale) if self.slope_scale > 0 else 0.0
        w_eff = 1.0 - math.exp(-s)   # s 越大 → w_eff 越接近 1
        w_eff = max(0.0, min(1.0, w_eff))
        return w_eff
    
    # def _compute_effective_weight(self):
    #     """
    #     根据最近 history_len 个 d 的线性回归斜率，计算 w_effective：
    #     - slope > 0 较大 → 说明难度整体还在变化 → w_effective 接近 1
    #     """
    #     hist = self.state.get("history", [])
    #     L = len(hist)
    #     if L < 5:
    #         # 数据太少，默认认为有训练价值
    #         return 1.0

    #     d_vals = hist
    #     # t = 0, 1, ..., L-1
    #     t_mean = (L - 1) / 2.0
    #     d_mean = sum(d_vals) / float(L)

    #     num = 0.0
    #     den = 0.0
    #     for i, d in enumerate(d_vals):
    #         dt = i - t_mean
    #         num += dt * (d - d_mean)
    #         den += dt * dt

    #     if den <= 1e-8:
    #         return 1.0

    #     slope = abs(num / den)
        
    #     # 缩放 + 平滑映射：slope≈slope_scale 时，w_effective 已经比较高
    #     s = slope / float(self.slope_scale) if self.slope_scale > 0 else 0.0
    #     w_eff = 1.0 - math.exp(-s)   # s 越大 → w_eff 越接近 1
    #     w_eff = max(0.0, min(1.0, w_eff))
    #     return w_eff

    # def _compute_effective_weight(self):
    #     """
    #     使用“前后半段难度均值差”的方案：
    #     - d_first = 前半段均值
    #     - d_second = 后半段均值
    #     - trend = d_second - d_first
    #     定义停滞得分：
    #         stagnation = 1 / (1 + |trend| / c_trend)
    #     其中 c_trend = self.slope_scale。
    #     最终有效权重：
    #         w_effective = 1 - stagnation
    #     trend 越接近 0（越停滞） → stagnation 越接近 1 → w_effective 越接近 0。
    #     trend 绝对值越大 → w_effective 越接近 1。
    #     """
    #     hist = self.state.get("history", [])
    #     L = len(hist)
    #     if L < 5:
    #         # 数据太少，默认认为有训练价值
    #         return 1.0

    #     d_vals = hist
    #     mid = L // 2
    #     d_first = sum(d_vals[:mid]) / float(mid)
    #     d_second = sum(d_vals[mid:]) / float(L - mid)

    #     trend = d_second - d_first

    #     c_trend = self.slope_scale
    #     if c_trend <= 1e-8:
    #         # 避免除零，退化为“有变化就有效”
    #         stagnation = 0.0 if abs(trend) > 0 else 1.0
    #     else:
    #         stagnation = 1.0 / (1.0 + abs(trend) / float(c_trend))
    #         # 理论上已经在 (0,1]，再保险裁剪一下
    #         stagnation = max(0.0, min(1.0, stagnation))

    #     w_eff = 1.0 - stagnation
    #     w_eff = max(0.0, min(1.0, w_eff))
    #     return w_eff

    # def _compute_effective_weight(self):
    #     """
    #     基于最近一段 history["d"] 计算环境的有效权重 w_eff ∈ [0, 1]。

    #     使用两个维度：
    #     1) 活动度（range_score）：
    #     - range_val = max(history) - min(history)
    #     - range_val 越大，说明这一段内尝试过的难度跨度越大，环境越“活跃”
    #     - 映射方式：range_score = 1 - exp(- range_val / range_scale)
    #         其中 range_scale 用 self.slope_scale 充当尺度参数

    #     2) 一致性（consistency_score）：
    #     - diffs = [d[i+1] - d[i]]，i = 0..L-2
    #     - std_diff = std(diffs)（相邻步之间难度变化的标准差）
    #     - std_diff 越小，说明调整步伐越稳定、不那么乱跳
    #     - 映射方式：consistency_score = exp(- std_diff / std_scale)
    #         其中 std_scale 控制对“震荡程度”的敏感度

    #     最终权重：
    #         w_eff = range_score * consistency_score

    #     直观含义：
    #         - 难度跨度大 ----> 变化比较稳定----> range_score 和 consistency_score 都偏高 → w_eff 偏高
    #         - 难度跨度小 & 慢慢增长 ----> range_score 低，一致性高 → w_eff 中等
    #         - 难度跨度小 & 震荡 ----> range_score低，一致性低 → w_eff 降低
    #         - 难度跨度0 & 不变化 ---->  range_score 0
    #     """
    #     hist = self.state.get("history", [])
    #     L = len(hist)
    #     if L < 5:
    #         return 1.0

    #     # ========= 1. 活动度：max - min =========
    #     d_min = min(hist)
    #     d_max = max(hist)
    #     range_val = d_max - d_min

    #     if range_val <= 0:
    #         range_score = 0.0
    #     else:
    #         range_scale = float(getattr(self, "slope_scale", 3.0))
    #         x = range_val / range_scale
    #         range_score = 1.0 - math.exp(-x)

    #     # ========= 2. 一致性：std(diff) 越小越好 =========
    #     diffs = [hist[i+1] - hist[i] for i in range(L - 1)]

    #     if len(diffs) <= 1:
    #         std_diff = 0.0
    #     else:
    #         mean_diff = sum(diffs) / len(diffs)
    #         var_diff = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
    #         std_diff = math.sqrt(var_diff)

    #     std_scale = float(getattr(self, "std_scale", 0.8))
    #     if std_scale <= 1e-8:
    #         std_scale = 0.8

    #     y = std_diff / std_scale
    #     consistency_score = math.exp(-y)

    #     # ========= 3. 合并两个分数 =========
    #     w_eff = range_score * consistency_score

    #     return w_eff

    def _compute_recency_weight(self, global_step):
        """
        未访问时间权重：
        - 从未访问过 → 1.0
        - age = global_step - last_step
        - w_rec = min(1, age / age_scale)
        """
        last_step = self.state.get("last_step", -1)
        if last_step is None or last_step < 0:
            return 1.0
        age = max(0, global_step - last_step)
        if self.age_scale <= 0:
            return 0.0
        w = age / float(self.age_scale)
        w = max(0.0, min(1.0, w))
        return w

    def get_weight(self, global_step):
        """
        给环境采样用的权重：
        weight = alpha * w_effective + beta * w_recency
        外面有一堆 DifficultyControl 时，你可以：
            weights = [dc.get_weight(global_step) for dc in controllers]
        然后归一化按权重采样环境。
        """
        w_eff = self._compute_effective_weight()
        w_rec = self._compute_recency_weight(global_step)
        weight = self.alpha * w_eff + self.beta * w_rec
        return max(0.0, float(weight))
    def _count_zero_correct(self):
        hist = self.state.get("correct_history",[])
        count = 0
        pos = len(hist) -1
        while pos > 0 and hist[pos]==0:
            count += 1
            pos -= 1
        return count
    def _count_max_difficulty(self):
        eps = 1e-7
        hist = self.state.get("distance_history",[])
        count = 0
        pos = len(hist) -1
        while pos > 0 and  abs(hist[pos]-self.dmax) <= eps :
            count += 1
            pos -= 1
        return count  
    def _compute_slope(self):
        """
        根据最近 history_len 个 d 的线性回归斜率，计算 w_effective：
        - slope > 0 较大 → 说明难度整体还在涨 → w_effective 接近 1
        - slope ≈ 0 或负 → 难度整体不涨（只是抖动） → w_effective 接近 0
        """
        hist = self.state.get("distance_history", [])
        L = len(hist)
        if L < self.history_len:
            return 1.0

        d_vals = hist
        t_mean = (L - 1) / 2.0
        d_mean = sum(d_vals) / float(L)

        num = 0.0
        den = 0.0
        for i, d in enumerate(d_vals):
            dt = i - t_mean
            num += dt * (d - d_mean)
            den += dt * dt

        if den <= 1e-8:
            return 1.0

        slope = num / den
        return slope
    
    def get_windows_state(self):
        zero_correct_count=self._count_zero_correct()
        max_distance_count = self._count_max_difficulty()
        distance_slope = self._compute_slope()
        
        return{
            "zero_correct_count":zero_correct_count,
            "max_distance_count":max_distance_count,
            "distance_slope":distance_slope
        }

    def empty_histroy(self):
        self.state['distance_history'] = []
        self.state['correct_history'] = []
    # ================== 序列化 ==================

    def _to_serializable(self):
        serializable_object = {
            "version": 2,  # 版本号改成 2，表示有 history 等新字段
            "params": {
                "target": self.target,
                "k": self.k,
                "dmin": self.dmin,
                "dmax": self.dmax,
                "step_cap": self.step_cap,
                "jitter": self.jitter,
                "ema_beta": self.ema_beta,
                "state": {k: v for k, v in self.state.items()},
                "activate_function": self.activate_function,
                "k_updater_name": self.k_updater_name,
                # 新增参数
                "history_len": self.history_len,
                "slope_scale": self.slope_scale,
                "age_scale": self.age_scale,
                "std_scale": self.std_scale,
                "alpha": self.alpha,
                "beta": self.beta,
            },
        }
        if self.k_updater_name != "fixed":
            serializable_object['k_updater'] = self.k_updater._to_serializable()
        
        return serializable_object
    
    @classmethod
    def _from_serializable(cls, payload):
        params = payload["params"]
        obj = cls(**params)
        if obj.k_updater_name != "fixed" and payload.get("k_updater"):
            k_updater = BetaScheduler._from_serializable(payload["k_updater"])
            obj.k_updater = k_updater
        # 兼容老版本：如果没有 history 字段，补一个
        if "distance_history" not in obj.state:
            obj.state["distance_history"] = []
        if "correct_history" not in obj.state:
            obj.state["correct_history"] = []
        return obj

    @staticmethod
    def json_default(o):
        if isinstance(o, DifficultyControl):
            return o._to_serializable()
        raise TypeError(f"{type(o)} is not JSON serializable")

    @staticmethod
    def json_object_hook(d):
        if isinstance(d, dict) and d.get("version") in (1, 2) and "params" in d:
            try:
                return DifficultyControl._from_serializable(d)
            except Exception as e:
                print(f"Error in json_object_hook, please check your json config ", e)
                return d
        return d



if __name__ == "__main__":
    control = DifficultyControl(dmax=10, ema_beta=0.0)
    # 测试序列化和反序列化
    json_str = json.dumps(control, default=DifficultyControl.json_default)
    print(json_str)
    
    # 从 JSON 恢复对象
    control_obj = json.loads(json_str, object_hook=DifficultyControl.json_object_hook)
    print(control_obj)
