struct KernelComputeForces {
  const int big = 5;
  const int small = -4;
  StencilInfo stencil{small, small, big, big, true};
  StencilInfo stencil2{small, small, big, big, true};
  const int bigg = _BS_ + big - 1;
  const Real c0 = -137. / 60.;
  const Real c1 = 5.;
  const Real c2 = -5.;
  const Real c3 = 10. / 3.;
  const Real c4 = -5. / 4.;
  const Real c5 = 1. / 5.;
  bool inrange(const int i) const { return (i >= small && i < bigg); }
  const std::vector<BlockInfo> &presInfo = var.pres->infos;
  void operator()(VectorLab &l, ScalarLab &chi, const BlockInfo &info,
                  const BlockInfo &info2) const {
    int nm = _BS_ + stencil.ex - stencil.sx - 1;
    Real *uchi = (Real *)chi.m;
    Real *um = (Real *)l.m;
    Real *P = (Real *)presInfo[info.id].block;
    for (auto &shape : sim.shapes) {
      std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      Real vel_norm = std::sqrt(shape->u * shape->u + shape->v * shape->v);
      Real vel_unit[2] = {vel_norm > 0 ? (Real)shape->u / vel_norm : (Real)0,
                          vel_norm > 0 ? (Real)shape->v / vel_norm : (Real)0};
      Real NUoH = sim.nu / info.h;
      ObstacleBlock *O = OBLOCK[info.id];
      if (O == nullptr)
        continue;
      assert(O->filled);
      for (size_t k = 0; k < O->n_surfPoints; ++k) {
        int ix = O->surface[k].ix, iy = O->surface[k].iy;
        Real p[2];
        p[0] = info.origin[0] + info.h * (ix + 0.5);
        p[1] = info.origin[1] + info.h * (iy + 0.5);
        Real normX = O->surface[k].dchidx;
        Real normY = O->surface[k].dchidy;
        Real norm = 1.0 / std::sqrt(normX * normX + normY * normY);
        Real dx = normX * norm;
        Real dy = normY * norm;
        Real DuDx;
        Real DuDy;
        Real DvDx;
        Real DvDy;
        int x = ix;
        int y = iy;
        for (int kk = 0; kk < 5; kk++) {
          int dxi = round(kk * dx);
          int dyi = round(kk * dy);
          if (ix + dxi + 1 >= _BS_ + big - 1 || ix + dxi - 1 < small)
            continue;
          if (iy + dyi + 1 >= _BS_ + big - 1 || iy + dyi - 1 < small)
            continue;
          x = ix + dxi;
          y = iy + dyi;
          int x0 = x - stencil.sx;
          int y0 = y - stencil.sy;
          if (uchi[nm * y0 + x0] < 0.01)
            break;
        }
        int sx = normX > 0 ? +1 : -1;
        int sy = normY > 0 ? +1 : -1;
	int x0 = x - stencil.sx;
	int y0 = y - stencil.sy;
	int ix0 = ix - stencil.sx;
	int iy0 = iy - stencil.sy;
        const Real *l00 = um + 2 * (nm * (y0) + x0);
        const Real *l01 = um + 2 * (x0 + nm * (y0) + sx);
        const Real *l02 = um + 2 * (x0 + nm * (y0) + 2 * sx);
        const Real *l03 = um + 2 * (x0 + nm * (y0) + 3 * sx);
        const Real *l04 = um + 2 * (x0 + nm * (y0) + 4 * sx);
        const Real *l05 = um + 2 * (x0 + nm * (y0) + 5 * sx);
        const Real *l06 = um + 2 * (nm * (y0 + sy) + x0);
        const Real *l07 = um + 2 * (nm * (y0 + 2 * sy) + x0);
        const Real *l08 = um + 2 * (nm * (y0 + 3 * sy) + x0);
        const Real *l09 = um + 2 * (nm * (y0 + 4 * sy) + x0);
        const Real *l10 = um + 2 * (nm * (y0 + 5 * sy) + x0);
        const Real *l11 = um + 2 * (nm * (y0) + x0 - 1);
        const Real *l12 = um + 2 * (x0 + nm * (y0) + 1);
        const Real *l13 = um + 2 * (nm * (y0 - 1) + x0);
        const Real *l14 = um + 2 * (nm * (y0 + 1) + x0);
        const Real *l15 = um + 2 * (x0 + nm * (y0 + sy) + 2 * sx);
        const Real *l16 = um + 2 * (x0 + nm * (y0 + 2 * sy) + 2 * sx);
        const Real *l17 = um + 2 * (x0 + nm * (y0 + sy) + sx);
        const Real *l18 = um + 2 * (x0 + nm * (y0 + 2 * sy) + sx);
        const Real *l19 = um + 2 * (nm * (iy0) + ix0);
	Real dveldx2[2], dveldy2[2], dveldxdy[2], dveldy[2], dveldx[2];
	for (int d = 0; d < 2; d++) {
	  if (inrange(x + 5 * sx))
	    dveldx[d] =
              sx * (c0 * (*(l00 + d)) + c1 * (*(l01 + d)) + c2 * (*(l02 + d)) +
                    c3 * (*(l03 + d)) + c4 * (*(l04 + d)) + c5 * (*(l05 + d)));
	  else if (inrange(x + 2 * sx))
	    dveldx[d] = sx * (-1.5 * (*(l00 + d)) + 2.0 * (*(l01 + d)) -
			   0.5 * (*(l02 + d)));
	  else
	    dveldx[d] = sx * ((*(l01 + d)) - (*(l00 + d)));
	  if (inrange(y + 5 * sy))
	    dveldy[d] =
              sy * (c0 * (*(l00 + d)) + c1 * (*(l06 + d)) + c2 * (*(l07 + d)) +
                    c3 * (*(l08 + d)) + c4 * (*(l09 + d)) + c5 * (*(l10 + d)));
	  else if (inrange(y + 2 * sy))
	    dveldy[d] = sy * (-1.5 * (*(l00 + d)) + 2.0 * (*(l06 + d)) -
			   0.5 * (*(l07 + d)));
	  else
	    dveldy[d] = sx * ((*(l06 + d)) - (*(l00 + d)));
	  dveldx2[d] = (*(l11 + d)) - 2.0 * (*(l00 + d)) + (*(l12 + d));
	  dveldy2[d] = (*(l13 + d)) - 2.0 * (*(l00 + d)) + (*(l14 + d));
	  
	  if (inrange(x + 2 * sx) && inrange(y + 2 * sy))
	    dveldxdy[d] = sx * sy *
	      (-0.5 * (-1.5 * (*(l02 + d)) + 2 * (*(l15 + d)) -
		       0.5 * (*(l16 + d))) +
	       2 * (-1.5 * (*(l01 + d)) + 2 * (*(l17 + d)) -
		    0.5 * (*(l18 + d))) -
	       1.5 * (-1.5 * (*(l00 + d)) + 2 * (*(l06 + d)) -
		      0.5 * (*(l07 + d))));
	  else
	    dveldxdy[d] = sx * sy * ((*(l17 + d)) - (*(l01 + d))) -
	      ((*(l06 + d)) - (*(l00 + d)));
	}
        DuDx = dveldx[0] + dveldx2[0] * (ix - x) + dveldxdy[0] * (iy - y);
        DvDx = dveldx[1] + dveldx2[1] * (ix - x) + dveldxdy[1] * (iy - y);
        DuDy = dveldy[0] + dveldy2[0] * (iy - y) + dveldxdy[0] * (ix - x);
        DvDy = dveldy[1] + dveldy2[1] * (iy - y) + dveldxdy[1] * (ix - x);
        Real fXV = NUoH * DuDx * normX + NUoH * DuDy * normY,
             fXP = -P[_BS_ * iy + ix] * normX;
        Real fYV = NUoH * DvDx * normX + NUoH * DvDy * normY,
             fYP = -P[_BS_ * iy + ix] * normY;
        Real fXT = fXV + fXP, fYT = fYV + fYP;
        O->x_s[k] = p[0];
        O->y_s[k] = p[1];
        O->p_s[k] = P[_BS_ * iy + ix];
        O->u_s[k] = *(l19 + 0);
        O->v_s[k] = *(l19 + 1);
        O->nx_s[k] = dx;
        O->ny_s[k] = dy;
        O->omega_s[k] = (DvDx - DuDy) / info.h;
        O->uDef_s[k] = O->udef[iy][ix][0];
        O->vDef_s[k] = O->udef[iy][ix][1];
        O->fX_s[k] = -P[_BS_ * iy + ix] * dx + NUoH * DuDx * dx + NUoH * DuDy * dy;
        O->fY_s[k] = -P[_BS_ * iy + ix] * dy + NUoH * DvDx * dx + NUoH * DvDy * dy;
        O->fXv_s[k] = NUoH * DuDx * dx + NUoH * DuDy * dy;
        O->fYv_s[k] = NUoH * DvDx * dx + NUoH * DvDy * dy;
        O->perimeter += std::sqrt(normX * normX + normY * normY);
        O->circulation += normX * O->v_s[k] - normY * O->u_s[k];
        O->forcex += fXT;
        O->forcey += fYT;
        O->forcex_V += fXV;
        O->forcey_V += fYV;
        O->forcex_P += fXP;
        O->forcey_P += fYP;
        O->torque += (p[0] - Cx) * fYT - (p[1] - Cy) * fXT;
        O->torque_P += (p[0] - Cx) * fYP - (p[1] - Cy) * fXP;
        O->torque_V += (p[0] - Cx) * fYV - (p[1] - Cy) * fXV;
        Real forcePar = fXT * vel_unit[0] + fYT * vel_unit[1];
        O->thrust += .5 * (forcePar + std::fabs(forcePar));
        O->drag -= .5 * (forcePar - std::fabs(forcePar));
        Real forcePerp = fXT * vel_unit[1] - fYT * vel_unit[0];
        O->lift += forcePerp;
        Real powOut = fXT * O->u_s[k] + fYT * O->v_s[k];
        Real powDef = fXT * O->uDef_s[k] + fYT * O->vDef_s[k];
        O->Pout += powOut;
        O->defPower += powDef;
        O->PoutBnd += std::min((Real)0, powOut);
        O->defPowerBnd += std::min((Real)0, powDef);
      }
      O->PoutNew = O->forcex * shape->u + O->forcey * shape->v;
    }
  }
};
