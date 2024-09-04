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
    ScalarBlock &P = *(ScalarBlock *)presInfo[info.id].block;
    for (auto &shape : sim.shapes) {
      std::vector<ObstacleBlock *> &OBLOCK = shape->obstacleBlocks;
      Real Cx = shape->centerOfMass[0], Cy = shape->centerOfMass[1];
      Real vel_norm =
          std::sqrt(shape->u * shape->u + shape->v * shape->v);
      Real vel_unit[2] = {
          vel_norm > 0 ? (Real)shape->u / vel_norm : (Real)0,
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
        Vector dveldx;
        if (inrange(x + 5 * sx))
          dveldx = sx * (c0 * l(x, y) + c1 * l(x + sx, y) +
                         c2 * l(x + 2 * sx, y) + c3 * l(x + 3 * sx, y) +
                         c4 * l(x + 4 * sx, y) + c5 * l(x + 5 * sx, y));
        else if (inrange(x + 2 * sx))
          dveldx = sx * (-1.5 * l(x, y) + 2.0 * l(x + sx, y) -
                         0.5 * l(x + 2 * sx, y));
        else
          dveldx = sx * (l(x + sx, y) - l(x, y));
        Vector dveldy;
        if (inrange(y + 5 * sy))
          dveldy = sy * (c0 * l(x, y) + c1 * l(x, y + sy) +
                         c2 * l(x, y + 2 * sy) + c3 * l(x, y + 3 * sy) +
                         c4 * l(x, y + 4 * sy) + c5 * l(x, y + 5 * sy));
        else if (inrange(y + 2 * sy))
          dveldy = sy * (-1.5 * l(x, y) + 2.0 * l(x, y + sy) -
                         0.5 * l(x, y + 2 * sy));
        else
          dveldy = sx * (l(x, y + sy) - l(x, y));
        Vector dveldx2 = l(x - 1, y) - 2.0 * l(x, y) + l(x + 1, y);
        Vector dveldy2 = l(x, y - 1) - 2.0 * l(x, y) + l(x, y + 1);
        Vector dveldxdy;
        if (inrange(x + 2 * sx) && inrange(y + 2 * sy))
          dveldxdy =
              sx * sy *
              (-0.5 * (-1.5 * l(x + 2 * sx, y) + 2 * l(x + 2 * sx, y + sy) -
                       0.5 * l(x + 2 * sx, y + 2 * sy)) +
               2 * (-1.5 * l(x + sx, y) + 2 * l(x + sx, y + sy) -
                    0.5 * l(x + sx, y + 2 * sy)) -
               1.5 * (-1.5 * l(x, y) + 2 * l(x, y + sy) -
                      0.5 * l(x, y + 2 * sy)));
        else
          dveldxdy = sx * sy * (l(x + sx, y + sy) - l(x + sx, y)) -
                     (l(x, y + sy) - l(x, y));
        DuDx = dveldx.u[0] + dveldx2.u[0] * (ix - x) + dveldxdy.u[0] * (iy - y);
        DvDx = dveldx.u[1] + dveldx2.u[1] * (ix - x) + dveldxdy.u[1] * (iy - y);
        DuDy = dveldy.u[0] + dveldy2.u[0] * (iy - y) + dveldxdy.u[0] * (ix - x);
        DvDy = dveldy.u[1] + dveldy2.u[1] * (iy - y) + dveldxdy.u[1] * (ix - x);
        Real fXV = NUoH * DuDx * normX + NUoH * DuDy * normY,
                   fXP = -P[iy][ix] * normX;
        Real fYV = NUoH * DvDx * normX + NUoH * DvDy * normY,
                   fYP = -P[iy][ix] * normY;
        Real fXT = fXV + fXP, fYT = fYV + fYP;
        O->x_s[k] = p[0];
        O->y_s[k] = p[1];
        O->p_s[k] = P[iy][ix];
        O->u_s[k] = l(ix, iy).u[0];
        O->v_s[k] = l(ix, iy).u[1];
        O->nx_s[k] = dx;
        O->ny_s[k] = dy;
        O->omega_s[k] = (DvDx - DuDy) / info.h;
        O->uDef_s[k] = O->udef[iy][ix][0];
        O->vDef_s[k] = O->udef[iy][ix][1];
        O->fX_s[k] = -P[iy][ix] * dx + NUoH * DuDx * dx + NUoH * DuDy * dy;
        O->fY_s[k] = -P[iy][ix] * dy + NUoH * DvDx * dx + NUoH * DvDy * dy;
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
