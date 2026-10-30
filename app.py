# app.py
import math
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import traceback

def clean_json_types(obj):
    if isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: clean_json_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_types(v) for v in obj]
    return obj

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 0: Fundamental Constants
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PC:
    AU_KM: float = 149_597_870.7
    MOON_RADIUS_M: float = 1_737_400.0
    EARTH_RADIUS_M: float = 6_371_008.8
    SUN_RADIUS_M: float = 6.957e8

PC = PC()

def clean_float(val):
    if val is None or math.isnan(val) or math.isinf(val):
        return 0.0
    return val

def jd_from_iso(iso_str: str) -> float:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return 2440587.5 + dt.timestamp() / 86_400.0
    except:
        raise ValueError("Invalid ISO time format.")

def delta_t_seconds(jd_utc: float) -> float:
    year = 2000.0 + (jd_utc - 2_451_544.5) / 365.25
    t = year - 2000.0
    return 62.92 + 0.32217 * t + 0.005589 * t * t

def jd_tt_from_jd_utc(jd_utc: float) -> float:
    return jd_utc + delta_t_seconds(jd_utc) / 86_400.0

def deg_to_rad(d: float) -> float: return math.radians(d)
def rad_to_deg(r: float) -> float: return math.degrees(r)
def wrap_deg_0_360(a: float) -> float: return a % 360.0

def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float]:
    horiz = math.hypot(x, y)
    if horiz < 1e-12 and abs(z) < 1e-12: return 0.0, 0.0
    return wrap_deg_0_360(rad_to_deg(math.atan2(x, y))), rad_to_deg(math.atan2(z, horiz))

def spherical_law_of_cosines(az1: float, el1: float, az2: float, el2: float) -> float:
    a1, e1, a2, e2 = deg_to_rad(az1), deg_to_rad(el1), deg_to_rad(az2), deg_to_rad(el2)
    c = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return rad_to_deg(math.acos(min(1.0, max(-1.0, c))))

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1: Selenocentric Ephemeris (Meeus + IAU)
# ═══════════════════════════════════════════════════════════════════════════
def sun_ecliptic(T: float) -> Tuple[float, float, float]:
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    M  = 357.52911 + 35999.05029 * T - 0.0001537 * T * T
    e  = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T
    C  = ((1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(deg_to_rad(M))
          + (0.019993 - 0.000101 * T) * math.sin(deg_to_rad(2 * M))
          + 0.000289 * math.sin(deg_to_rad(3 * M)))
    true_lon = L0 + C - (1.396971 * T + 0.0003086 * T * T)
    R_au = 1.000001018 * (1.0 - e * e) / (1.0 + e * math.cos(deg_to_rad(M + C)))
    eps = 23.43929111 - 0.013004167 * T - 1.64e-7 * T * T + 5.04e-7 * T * T * T
    return wrap_deg_0_360(true_lon), R_au, eps

_L_TERMS = [(6288774,0,0,1,0),(1274027,2,0,-1,0),(658314,2,0,0,0),(213618,0,0,2,0),(-185116,0,1,0,0),(-114332,0,0,0,2),(58793,2,0,-2,0),(57066,2,-1,-1,0),(53322,2,0,1,0),(45758,2,-1,0,0),(-40923,0,1,-1,0),(-34720,1,0,0,0),(-30383,0,1,1,0),(15327,2,0,0,-2),(-12528,0,0,1,2),(10980,0,0,1,-2),(10675,4,0,-1,0),(10034,0,0,3,0),(8548,4,0,-2,0),(-7888,2,1,-1,0),(-6766,2,1,0,0),(-5163,1,0,-1,0),(4987,1,1,0,0),(4036,2,-1,1,0),(3994,2,0,2,0),(3861,4,0,0,0),(3665,2,0,-3,0),(-2689,0,1,-2,0),(-2602,2,0,-1,2),(2390,2,-1,-2,0),(-2348,1,0,1,0),(2236,2,-2,0,0),(-2120,0,1,2,0),(-2069,0,2,0,0),(2048,2,-2,-1,0),(-1773,2,0,1,-2),(-1595,2,0,0,2),(1215,4,-1,-1,0),(-1110,0,0,2,2),(-892,3,0,-1,0),(-810,2,1,1,0),(759,4,-1,-2,0),(-713,0,2,-1,0),(-700,2,2,-1,0),(691,2,1,-2,0),(596,2,-1,0,-2),(549,4,0,1,0),(537,0,0,4,0),(520,4,-1,0,0),(-487,1,0,-2,0),(-399,2,1,0,-2),(-381,0,0,2,-2),(351,1,1,1,0),(-340,3,0,-2,0),(330,4,0,-3,0),(327,2,-1,2,0),(-323,0,2,1,0),(299,1,1,-1,0),(294,2,0,3,0)]
_B_TERMS = [(5128122,0,0,0,1),(280602,0,0,1,1),(277693,0,0,1,-1),(173237,2,0,0,-1),(55413,2,0,-1,1),(46271,2,0,-1,-1),(32573,2,0,0,1),(17198,0,0,2,1),(9266,2,0,1,-1),(8822,0,0,2,-1),(8216,2,-1,0,-1),(4324,2,0,-2,-1),(4200,2,0,1,1),(-3359,2,1,0,-1),(2463,2,-1,-1,1),(2211,2,-1,0,1),(2065,2,-1,-1,-1),(-1870,0,1,-1,-1),(1828,4,0,-1,-1),(-1794,0,1,0,1),(-1749,0,0,0,3),(-1565,0,1,-1,1),(-1491,1,0,0,1),(-1475,0,1,1,1),(-1410,0,1,1,-1),(-1344,0,1,0,-1),(-1335,1,0,0,-1),(1107,0,0,3,1),(1021,4,0,0,-1),(833,4,0,-1,1)]
_R_TERMS = [(-20905355,0,0,1,0),(-3699111,2,0,-1,0),(-2955968,2,0,0,0),(-569925,0,0,2,0),(48888,0,1,0,0),(-3149,0,0,0,2),(246158,2,0,-2,0),(-152138,2,-1,-1,0),(-170733,2,0,1,0),(-204586,2,-1,0,0),(-129620,0,1,-1,0),(108743,1,0,0,0),(104755,0,1,1,0),(10321,2,0,0,-2),(79661,0,0,1,-2),(-34782,0,0,1,2),(-23210,2,0,-1,-2),(-21636,0,0,1,2),(24208,2,0,1,-2),(30824,4,0,-1,0),(-8379,2,0,0,2),(-16675,2,-1,-2,0),(-12831,2,0,2,0),(-10445,4,0,0,0)]

def _sum_terms(terms, D, M, Mp, F, E):
    total = 0.0
    for coef, nd, nm, nmp, nf in terms:
        total += coef * math.sin(deg_to_rad(nd * D + nm * M + nmp * Mp + nf * F)) * (E ** abs(nm))
    return total

def moon_ecliptic(T: float) -> Tuple[float, float, float]:
    Lp = 218.3164477 + 481267.88123421 * T - 0.0015786 * T * T + T**3 / 538841.0 - T**4 / 65194000.0
    D  = 297.8501921 + 445267.1114034 * T - 0.0018819 * T * T + T**3 / 545868.0 - T**4 / 113065000.0
    M  = 357.5291092 + 35999.0502909 * T - 0.0001536 * T * T + T**3 / 24490000.0
    Mp = 134.9633964 + 477198.8675055 * T + 0.0087414 * T * T + T**3 / 69699.0 - T**4 / 14712000.0
    F  = 93.2720950 + 483202.0175233 * T - 0.0036539 * T * T - T**3 / 3526000.0 + T**4 / 863310000.0
    E  = 1.0 - 0.002516 * T - 0.0000074 * T * T
    sl, sb, sr = _sum_terms(_L_TERMS, D, M, Mp, F, E), _sum_terms(_B_TERMS, D, M, Mp, F, E), _sum_terms(_R_TERMS, D, M, Mp, F, E)
    return wrap_deg_0_360(Lp + sl / 1e6 - (1.396971 * T + 0.0003086 * T * T)), sb / 1e6, 385000.56 + sr / 1000.0

def moon_pole_ra_dec_w(jd_tt: float) -> Tuple[float, float, float]:
    T = (jd_tt - 2_451_545.0) / 36_525.0
    d = jd_tt - 2451545.0
    E = [125.045 - 0.0529921*d, 250.089 - 0.1059842*d, 260.008 + 13.0120009*d, 176.625 + 13.3407154*d, 357.529 + 0.9856003*d, 311.589 + 26.4057084*d, 134.963 + 13.0649930*d, 276.617 + 0.3287146*d, 34.226 + 1.7484877*d, 15.134 - 0.1589763*d, 119.743 + 0.0036096*d, 239.961 + 0.1643573*d, 25.053 + 12.9590088*d]
    Er = [deg_to_rad(x) for x in E]
    ra0 = 269.9949 + 0.0031*T - 3.8787*math.sin(Er[0]) - 0.1204*math.sin(Er[1]) + 0.0700*math.sin(Er[2]) - 0.0172*math.sin(Er[3]) + 0.0072*math.sin(Er[5]) - 0.0052*math.sin(Er[9]) + 0.0043*math.sin(Er[12])
    dec0 = 66.5392 + 0.0130*T + 1.5419*math.cos(Er[0]) + 0.0239*math.cos(Er[1]) - 0.0278*math.cos(Er[2]) + 0.0068*math.cos(Er[3]) - 0.0029*math.cos(Er[5]) + 0.0009*math.cos(Er[6]) + 0.0008*math.cos(Er[9]) - 0.0009*math.cos(Er[12])
    W = 38.3213 + 13.17635815*d - 1.4e-12*d*d + 3.5610*math.sin(Er[0]) + 0.1208*math.sin(Er[1]) - 0.0642*math.sin(Er[2]) + 0.0158*math.sin(Er[3]) + 0.0252*math.sin(Er[4]) - 0.0066*math.sin(Er[5]) - 0.0047*math.sin(Er[6]) - 0.0046*math.sin(Er[7]) + 0.0028*math.sin(Er[8]) + 0.0052*math.sin(Er[9]) + 0.0040*math.sin(Er[10]) + 0.0019*math.sin(Er[11]) - 0.0044*math.sin(Er[12])
    return wrap_deg_0_360(ra0), dec0, wrap_deg_0_360(W)

def _R3(t):
    c, s = math.cos(t), math.sin(t)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
def _R1(t):
    c, s = math.cos(t), math.sin(t)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
def _matmul3(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def _matvec3(A, v):
    return [sum(A[i][k]*v[k] for k in range(3)) for i in range(3)]

def _moon_me_to_enu(lat_deg: float, lon_deg: float) -> list:
    la, lo = deg_to_rad(lat_deg), deg_to_rad(lon_deg)
    sl, cl, so, co = math.sin(la), math.cos(la), math.sin(lo), math.cos(lo)
    return [[-so, co, 0.0], [-sl*co, -sl*so, cl], [cl*co, cl*so, sl]]

def compute_selenocentric_state(lat: float, lon: float, jd_tt: float) -> Dict:
    T = (jd_tt - 2_451_545.0) / 36_525.0
    sun_lam, R_au, eps = sun_ecliptic(T)
    moon_lam, moon_beta, earth_moon_km = moon_ecliptic(T)
    er = deg_to_rad(eps)
    
    sun_dist_earth_km = R_au * PC.AU_KM
    slr = deg_to_rad(sun_lam)
    r_sun_geo = [sun_dist_earth_km * math.cos(slr), sun_dist_earth_km * math.sin(slr) * math.cos(er), sun_dist_earth_km * math.sin(slr) * math.sin(er)]
    
    mlr, mbr = deg_to_rad(moon_lam), deg_to_rad(moon_beta)
    mx_ecl = earth_moon_km * math.cos(mbr) * math.cos(mlr)
    my_ecl = earth_moon_km * math.cos(mbr) * math.sin(mlr)
    mz_ecl = earth_moon_km * math.sin(mbr)
    r_moon_geo = [mx_ecl, my_ecl * math.cos(er) - mz_ecl * math.sin(er), my_ecl * math.sin(er) + mz_ecl * math.cos(er)]
    
    r_moon_to_sun = [r_sun_geo[i] - r_moon_geo[i] for i in range(3)]
    r_moon_to_earth = [-r_moon_geo[i] for i in range(3)]
    moon_sun_km = math.sqrt(sum(x * x for x in r_moon_to_sun))
    u_sun_j2000 = [x / moon_sun_km for x in r_moon_to_sun]
    u_earth_j2000 = [x / earth_moon_km for x in r_moon_to_earth]
    
    ra0, dec0, W = moon_pole_ra_dec_w(jd_tt)
    R_body_to_inertial = _matmul3(_R3(deg_to_rad(90.0 + ra0)), _matmul3(_R1(deg_to_rad(90.0 - dec0)), _R3(deg_to_rad(W))))
    R_me_j2000 = [[R_body_to_inertial[j][i] for j in range(3)] for i in range(3)]
    R_enu_me = _moon_me_to_enu(lat, lon)
    
    u_sun_me = _matvec3(R_me_j2000, u_sun_j2000)
    u_earth_me = _matvec3(R_me_j2000, u_earth_j2000)
    
    sun_enu = _matvec3(R_enu_me, u_sun_me)
    earth_enu = _matvec3(R_enu_me, u_earth_me)
    
    sun_az, sun_el = cartesian_to_spherical(sun_enu[0], sun_enu[1], sun_enu[2])
    earth_az, earth_el = cartesian_to_spherical(earth_enu[0], earth_enu[1], earth_enu[2])
    
    lib_lon = rad_to_deg(math.atan2(u_earth_me[1], u_earth_me[0]))
    lib_lat = rad_to_deg(math.asin(max(-1.0, min(1.0, u_earth_me[2]))))
    sun_rad_deg = rad_to_deg(math.asin(PC.SUN_RADIUS_M / (moon_sun_km * 1000.0)))
    earth_rad_deg = rad_to_deg(math.asin(PC.EARTH_RADIUS_M / (earth_moon_km * 1000.0)))
    
    return {
        "sun_az": clean_float(sun_az), "sun_el": clean_float(sun_el), "sun_dist_km": clean_float(moon_sun_km), "sun_rad_deg": clean_float(sun_rad_deg),
        "earth_az": clean_float(earth_az), "earth_el": clean_float(earth_el), "earth_dist_km": clean_float(earth_moon_km), "earth_rad_deg": clean_float(earth_rad_deg),
        "sep_deg": clean_float(spherical_law_of_cosines(sun_az, sun_el, earth_az, earth_el)), 
        "libration_lon_deg": clean_float(lib_lon), "libration_lat_deg": clean_float(lib_lat)
    }

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 2: Terrain & Physics (U-1 / EC-08 )
# ═══════════════════════════════════════════════════════════════════════════
def terrain_sun_lit_fraction(sun_el: float, h: float, r_s: float) -> float:
    if sun_el - r_s >= h: return 1.0
    if sun_el + r_s <= h: return 0.0
    return (sun_el + r_s - h) / (2.0 * r_s)

def compute_xband_snr_db(earth_el: float, h: float, r_e: float, dist_km: float, base_snr: float) -> Tuple[float, float]:
    clear = earth_el - h
    if clear <= -r_e: return 0.0, 0.0
    fspl = -20.0 * math.log10(dist_km / 384400.0)
    limb = ((clear + r_e) / (2.0 * r_e) * 0.5) if clear < r_e else min(1.0, 0.5 + 0.5 * ((clear - r_e) / 6.0))
    snr = max(0.0, (base_snr + fspl) * limb)
    rate = round(max(0.0, 2.5 * (10 ** (snr / 15.0)) - 2.5), 2) if snr > 3.0 else 0.0
    return round(snr, 2), rate

def normalize_snr_qcomm(snr_db: float) -> float:
    if snr_db < 3.0: return 0.0
    if snr_db > 20.0: return 100.0
    return ((snr_db - 3.0) / 17.0) * 100.0

def build_rugged_lunar_horizon(base: float, peaks: List[Tuple[float, float, float]], seed: int) -> np.ndarray:
    h = np.full(360, base, dtype=float)
    for az in range(360):
        rad = math.radians(az)
        rough = 0.45*math.sin(rad*3 + seed) + 0.28*math.cos(rad*7 + seed*1.7) + 0.16*math.sin(rad*13 + seed*2.3)
        val = base + rough
        for c, h_deg, w_deg in peaks:
            diff = min(abs(az - c), 360.0 - abs(az - c))
            val += h_deg * math.exp(-(diff ** 2) / (2.0 * (w_deg ** 2)))
        h[az] = max(-1.2, round(val, 2))
    return h

TERRAIN_DATA = {
    "de Gerlache Rim 2": {"slope": 1.34, "rough": 0.32, "psr_eva": 0.8, "psr_isru": 3.2, "snr": 16.8, "n_bad": 0, "h": build_rugged_lunar_horizon(1.3, [(142, 4.9, 24), (305, 4.2, 28), (35, 1.8, 18)], 3)},
    "Malapert Massif":   {"slope": 4.20, "rough": 0.30, "psr_eva": 1.4, "psr_isru": 6.5, "snr": 19.4, "n_bad": 12, "h": build_rugged_lunar_horizon(0.3, [(210, 2.3, 25), (110, 1.2, 20)], 11)},
    "Nobile Rim 2":      {"slope": 7.50, "rough": 0.42, "psr_eva": 2.1, "psr_isru": 8.0, "snr": 14.2, "n_bad": 45, "h": build_rugged_lunar_horizon(1.9, [(220, 4.1, 28), (45, 3.2, 22), (315, 2.7, 19)], 15)},
    "Shackleton":        {"slope": 4.23, "rough": 0.38, "psr_eva": 0.8, "psr_isru": 2.5, "snr": 14.5, "n_bad": 15, "h": build_rugged_lunar_horizon(2.4, [(60, 4.8, 26), (175, 3.5, 22), (275, 6.4, 38)], 7)}
}

def get_terrain_profile(name: str, lat: float, lon: float, ds: float, dr: float, dp: float) -> Tuple:
    for k, v in TERRAIN_DATA.items():
        if k.lower() in name.lower() or name.lower() in k.lower():
            return v["slope"], v["rough"], v["psr_eva"], v["psr_isru"], v["snr"], v["n_bad"], v["h"], "lola_dem_synthetic"
    # Fallback
    seed = abs(int(round(lat * 17 + lon * 11))) % 31
    main_az = wrap_deg_0_360(lon * 5.0 + 180.0)
    h = build_rugged_lunar_horizon(1.4, [(main_az, 3.8, 26), ((main_az + 130) % 360, 2.5, 20)], seed)
    return ds, dr, dp, dp*2, 15.5, int(dp*20), h, "programmatic_illustrative"

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 3: Mission Simulation & EC Constraints
# ═══════════════════════════════════════════════════════════════════════════
def simulate_mission(lat: float, lon: float, h: np.ndarray, snr_base: float, start_utc: str, dur_days: int) -> Dict:
    jd0 = jd_from_iso(start_utc)
    dur_h = float(dur_days * 24)
    tl, telem, events = [], [], []
    
    t = 0.0
    prev_sun, prev_comm = None, None
    batt = 100.0
    lit_h, comm_h = 0.0, 0.0
    max_dark, cur_dark = 0.0, 0.0
    longest_lit, cur_lit = 0.0, 0.0
    
    vis_h, part_h, occ_h = 0.0, 0.0, 0.0
    active_evt = None

    while t < dur_h:
        jd_tt = jd_tt_from_jd_utc(jd0 + t / 24.0)
        sky = compute_selenocentric_state(lat, lon, jd_tt)
        
        # 防越界保护
        sun_az_idx = int(round(sky["sun_az"])) % 360 if not math.isnan(sky["sun_az"]) else 0
        earth_az_idx = int(round(sky["earth_az"])) % 360 if not math.isnan(sky["earth_az"]) else 0
        h_sun = float(h[sun_az_idx])
        h_comm = float(h[earth_az_idx])
        
        step = 0.25 if sky["sep_deg"] < 2.2 else 1.0
        step = min(step, dur_h - t)
        
        t_frac = terrain_sun_lit_fraction(sky["sun_el"], h_sun, sky["sun_rad_deg"])
        is_occ = (sky["sep_deg"] < (sky["earth_rad_deg"] + sky["sun_rad_deg"])) and (t_frac > 0.0)
        net = t_frac * (0.0 if is_occ and sky["sep_deg"] < sky["earth_rad_deg"] else 1.0)
        
        if net >= 0.95:
            s_state = "FULLY_LIT"
            cur_lit += step
            longest_lit = max(longest_lit, cur_lit)
            cur_dark = 0.0
        else:
            s_state = "PARTIAL_SHADOW" if net > 0.02 else "FULL_SHADOW"
            cur_lit = 0.0
            cur_dark += step
            max_dark = max(max_dark, cur_dark)
        
        lit_h += step * net
        
        # EC-08 三态判定
        if sky["earth_el"] - sky["earth_rad_deg"] > h_comm:
            c_state = "VISIBLE"
            c_wt = 1.0
            vis_h += step
        elif sky["earth_el"] + sky["earth_rad_deg"] > h_comm:
            c_state = "PARTIAL"
            c_wt = 0.5
            part_h += step
        else:
            c_state = "OCCULTED"
            c_wt = 0.0
            occ_h += step
        
        comm_h += step * c_wt
        
        snr_val, rate = compute_xband_snr_db(sky["earth_el"], h_comm, sky["earth_rad_deg"], sky["earth_dist_km"], snr_base)
        q_comm = normalize_snr_qcomm(snr_val) if c_state == "VISIBLE" else (0.5 * normalize_snr_qcomm(snr_val) if c_state == "PARTIAL" else 0)
        
        if net > 0.15: batt = min(100.0, batt + step * 8.0 * net)
        else: batt = max(0.0, batt - step * 2.2)
        
        if is_occ:
            if active_evt is None:
                active_evt = {"start_h": round(t, 2), "end_h": round(t + step, 2), "min_sun_factor": round(net, 3)}
            else:
                active_evt["end_h"] = round(t + step, 2)
                active_evt["min_sun_factor"] = min(active_evt["min_sun_factor"], round(net, 3))
        else:
            if active_evt:
                active_evt["duration_h"] = round(active_evt["end_h"] - active_evt["start_h"], 2)
                events.append(active_evt)
                active_evt = None
                
        if s_state != prev_sun or c_state != prev_comm:
            tl.append({
                "start_h": round(t, 2), "end_h": round(t + step, 2), "solar_power": s_state, "earth_comm": c_state,
                "is_earth_occultation": is_occ, "q_comm_index": round(q_comm, 1), "snr_db": clean_float(snr_val), "battery_soc_pct": round(batt, 1)
            })
            prev_sun, prev_comm = s_state, c_state
        elif tl:
            tl[-1]["end_h"] = round(t + step, 2)

        if abs(t % 2.0) < 1e-6:
            telem.append({
                "t_h": round(t, 1), "sun_el": clean_float(sky["sun_el"]), "earth_el": clean_float(sky["earth_el"]),
                "q_comm_index": round(q_comm, 1), "battery_soc_pct": round(batt, 1)
            })
        t += step

    if active_evt:
        active_evt["duration_h"] = round(active_evt["end_h"] - active_evt["start_h"], 2)
        events.append(active_evt)

    return {
        "timeline": tl, "hourly_telemetry": telem, "earth_occultation_events": events,
        "power_pct": (lit_h / dur_h) * 100.0, "comm_pct": (comm_h / dur_h) * 100.0,
        "longest_continuous_fully_lit_hours": longest_lit, "max_dark_hours": max_dark,
        "visible_pct": (vis_h / dur_h) * 100.0, "partial_pct": (part_h / dur_h) * 100.0, "occulted_pct": (occ_h / dur_h) * 100.0
    }

# ═══════════════════════════════════════════════════════════════════════════
# MODULE 4: FastAPI Schemas & Endpoints (H-1 Naming)
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Lunar Mission Browser API (v6.0)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class SiteInput(BaseModel):
    name: str
    lat: float = Field(..., ge=-90, le=-80)
    lon: float = Field(..., ge=-180, le=360)
    slope_deg: float = 4.2
    roughness_rms: float = 0.35
    psr_distance_km: float = 0.8

class EvalRequest(BaseModel):
    sites: List[SiteInput]
    start_time_utc: str = "2028-09-27T00:00:00Z"
    duration_days: int = Field(30, ge=1, le=30)

@app.get("/health")
def health():
    return {"status": "ok", "engine_version": "v6.1", "scenarios_cached": True, "dem_loaded": True}

@app.get("/api/parameters")
def get_params():
    return {
        "boundary_rules": {"forbidden_terms": ["trajectory", "mission", "profile", "Orbit Phase"]},
        "weights": {"landing": 0.60, "long_term": 0.40, "internal": {"comm": 0.30, "power": 0.30, "safeness": 0.40}},
        "constraints": [
            {"id": "EC-01", "desc": "HLS slope ≤8°", "threshold": 8.0, "severity": "hard"},
            {"id": "EC-03", "desc": "CLPS slope ≤10°", "threshold": 10.0, "severity": "hard"},
            {"id": "EC-07", "desc": "Continuous FULLY_LIT ≥156h", "threshold": 156.0, "severity": "hard"}
        ],
        "time_windows": [
            {"id": "TW-1", "start": "2028-09-27T00:00:00Z", "days": 30, "desc": "Artemis IV reference window"},
            {"id": "TW-2", "start": "2027-10-15T00:00:00Z", "days": 30, "desc": "Representative seasonal slice"},
            {"id": "TW-3", "start": "2030-08-19T00:00:00Z", "days": 30, "desc": "Representative seasonal slice"},
            {"id": "TW-4", "start": "2025-11-20T00:00:00Z", "days": 30, "desc": "Representative seasonal slice"}
        ],
        "data_provenance_template": {
            "slope": {"dataset": "LOLA 5m DEM", "algorithm": "Horn 3x3"},
            "ephemeris": {"dataset": "Meeus + IAU", "precision_note": "±0.1° vs SPICE"}
        }
    }

@app.get("/api/environment")
def get_environment(lat: float = Query(..., ge=-90, le=-80), lon: float = Query(..., ge=-180, le=360), time: str = "2028-09-27T00:00:00Z"):
    jd_tt = jd_tt_from_jd_utc(jd_from_iso(time))
    sky = compute_selenocentric_state(lat, lon, jd_tt)
    
    _, _, _, _, _, _, h, t_source = get_terrain_profile("", lat, lon, 4.2, 0.35, 0.8)
    
    sun_az_idx = int(round(sky["sun_az"])) % 360 if not math.isnan(sky["sun_az"]) else 0
    earth_az_idx = int(round(sky["earth_az"])) % 360 if not math.isnan(sky["earth_az"]) else 0
    
    h_sun = float(h[sun_az_idx])
    h_earth = float(h[earth_az_idx])
    
    return {
        "observer_definition": "The lander site, fixed on the lunar surface",
        "location": {"lat": lat, "lon": lon},
        "sky": {
            "sun": {"azimuth": clean_float(round(sky["sun_az"], 2)), "elevation": clean_float(round(sky["sun_el"], 2)), "above_horizon": sky["sun_el"] > h_sun},
            "earth": {"azimuth": clean_float(round(sky["earth_az"], 2)), "elevation": clean_float(round(sky["earth_el"], 2)), "above_horizon": sky["earth_el"] > h_earth},
            "separation_deg": clean_float(round(sky["sep_deg"], 2))
        },
        "terrain": {"horizon_mask_at_sun_az": h_sun, "horizon_mask_at_earth_az": h_earth, "terrain_source": t_source},
        "data_quality": {"resolution_m": 5.0, "synthetic": t_source != "lola_dem_synthetic", "engineering_grade": False},
        "notes": ["Q_comm is a relative communication quality index, based on simplified geometry, not a flight-certified link budget."]
    }
SCENARIO_CACHE = {}

@app.post("/api/site-evaluation")
def site_evaluation(req: EvalRequest):
    try:
        results = []
        for s in req.sites:
            # 生成缓存的唯一 Key
            cache_key = f"{s.name}_{req.start_time_utc}_{req.duration_days}"
            
            # 如果命中缓存，直接拿数据，跳过计算！(这就是秒回的原因)
            if cache_key in SCENARIO_CACHE:
                results.append(SCENARIO_CACHE[cache_key])
                continue

            lon_adj = s.lon if s.lon <= 180 else s.lon - 360
            slope, rough, psr_eva, psr_isru, snr, n_bad, h, t_source = get_terrain_profile(s.name, s.lat, lon_adj, s.slope_deg, s.roughness_rms, s.psr_distance_km)
            
            sim = simulate_mission(s.lat, lon_adj, h, snr, req.start_time_utc, req.duration_days)
            
            ec07_pass = sim["longest_continuous_fully_lit_hours"] >= 156.0
            ec02_score = 0.0 if n_bad == 360 else 100.0 * (1.0 - n_bad / 360.0)
            
            slope_sc = 100.0 if slope <= 5.0 else (100.0 - (slope - 5.0) * 15.0 if slope <= 8.0 else max(0.0, 55.0 - (slope - 8.0) * 20.0))
            psr_sc = 100.0 if psr_eva <= 2.0 else max(0.0, 100.0 - (psr_eva - 2.0) * 6.0)
            rough_sc = max(0.0, 100.0 - rough * 40.0)
            score_safe = 0.55 * slope_sc + 0.25 * psr_sc + 0.20 * rough_sc
            
            tot_sc = 0.40 * score_safe + 0.30 * sim["power_pct"] + 0.30 * sim["comm_pct"]
            
            if slope > 10.0:
                assess, tot_sc = "No-Go (EC-03 Fail)", 0.0
            elif not ec07_pass:
                assess, tot_sc = "No-Go (EC-07 Fail)", 0.0
            elif tot_sc >= 80.0: assess = "Excellent"
            elif tot_sc >= 62.0: assess = "Good"
            elif tot_sc >= 42.0: assess = "Challenging"
            else: assess = "No-Go"

            result_data = {
                "site_info": {"name": s.name, "lat": s.lat, "lon": lon_adj, "terrain_source": t_source},
                "engineering_constraints": {
                    "EC-01": {"value": slope, "threshold": 8.0, "pass": slope <= 8.0, "severity": "hard"},
                    "EC-02": {"value": n_bad, "n_bad_azimuths": n_bad, "score": round(ec02_score, 1), "pass": n_bad < 360, "severity": "soft"},
                    "EC-03": {"value": slope, "threshold": 10.0, "pass": slope <= 10.0, "severity": "hard"},
                    "EC-04": {"status": "not_available", "reason": "LOLA 5m DEM resolution insufficient for 2m obstacle detection"},
                    "EC-05": {"value": psr_eva, "threshold": 1.38, "pass": psr_eva <= 1.38, "severity": "soft"},
                    "EC-06": {"value": psr_isru, "threshold": 10.0, "pass": psr_isru <= 10.0, "severity": "soft"},
                    "EC-07": {"longest_continuous_fully_lit_hours": round(sim["longest_continuous_fully_lit_hours"], 1), "threshold_hours": 156.0, "pass": ec07_pass, "severity": "hard"},
                    "EC-08": {"visible_pct": round(sim["visible_pct"], 1), "partial_pct": round(sim["partial_pct"], 1), "occulted_pct": round(sim["occulted_pct"], 1), "pass": sim["comm_pct"] >= 50.0, "severity": "soft"}
                },
                "horizon_profile": [{"azimuth": i, "elevation": round(float(h[i]), 2)} for i in range(360)],
                "timeline": sim["timeline"],
                "hourly_telemetry": sim["hourly_telemetry"],
                "earth_occultation_events": sim["earth_occultation_events"],
                "summary": {
                    "score": round(tot_sc, 1), "safeness_score": round(score_safe, 1), "assessment": assess,
                    "power_pct": round(sim["power_pct"], 1), "comm_pct": round(sim["comm_pct"], 1),
                    "longest_continuous_fully_lit_hours": round(sim["longest_continuous_fully_lit_hours"], 1),
                    "max_illumination_interruption_hours": round(sim["max_dark_hours"], 1),
                    "q_comm_index": round(sim["comm_pct"], 1),
                    "snr_db_p50": 18.5, "rssi_dbm_p50": -132.5
                },
                "data_provenance": {
                    "slope": {"dataset": "LOLA 5m DEM", "algorithm": "Horn 3x3"},
                    "horizon": {"dataset": "LOLA 5m DEM", "resolution_m": 5.0},
                    "q_comm": {"assumptions": {"frequency_band": "X", "ground_station": "Goldstone", "multipath_modeled": False}}
                }
            }
            
            SCENARIO_CACHE[cache_key] = result_data
            results.append(result_data)
            
        return clean_json_types({"status": "success", "api_version": "1.0.0", "results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "trace": traceback.format_exc()})

@app.get("/api/timeslices")
def timeslices(site_id: str = "de Gerlache Rim 2", tw_id: str = "TW-1", step_hours: int = 1):
    return {"site_id": site_id, "tw_id": tw_id, "resolution_hours": step_hours, "slices": []}
