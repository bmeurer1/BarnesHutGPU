/*
CUDA BarnesHut v3.1: Simulation of the gravitational forces
in a galactic cluster using the Barnes-Hut n-body algorithm

Copyright (c) 2013, Texas State University-San Marcos. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided that
the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, 
     this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University-San Marcos nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University-San Marcos <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher <burtscher@txstate.edu>
*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>
#include <time.h>

#ifdef __KEPLER__

// thread count
#define THREADS1 1024  /* must be a power of 2 */
#define THREADS2 1024
#define THREADS3 768
#define THREADS4 128
#define THREADS5 1024
#define THREADS6 1024

// block count = factor * #SMs
#define FACTOR1 2
#define FACTOR2 2
#define FACTOR3 1  /* must all be resident at the same time */
#define FACTOR4 4  /* must all be resident at the same time */
#define FACTOR5 2
#define FACTOR6 2

#else

// thread count
#define THREADS1 512  /* must be a power of 2 */
#define THREADS2 512
#define THREADS3 128
#define THREADS4 64
#define THREADS5 256
#define THREADS6 1024
#define THREADS7 256

// block count = factor * #SMs
#define FACTOR1 3
#define FACTOR2 3
#define FACTOR3 6  /* must all be resident at the same time */
#define FACTOR4 6  /* must all be resident at the same time */
#define FACTOR5 5
#define FACTOR6 1
#define FACTOR7 1

#endif

#define WARPSIZE 32
#define MAXDEPTH 32

__device__ volatile int stepd, bottomd, maxdepthd;
__device__ unsigned int blkcntd;
__device__ volatile double radiusd;

int h_stepd, h_maxdepthd;
double h_radiusd;


/******************************************************************************/
/*** initialize memory ********************************************************/
/******************************************************************************/

__global__ void InitializationKernel(int * __restrict errd)
{
  *errd = 0;
  stepd = -1;
  maxdepthd = 1;
  blkcntd = 0;
}


/******************************************************************************/
/*** compute center and radius ************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS1, FACTOR1)
void BoundingBoxKernel(int nnodesd, int nbodiesd, volatile int * __restrict startd, volatile int * __restrict childd, volatile double * __restrict massd, volatile double * __restrict posxd, volatile double * __restrict posyd, volatile double * __restrict poszd, volatile double * __restrict maxxd, volatile double * __restrict maxyd, volatile double * __restrict maxzd, volatile double * __restrict minxd, volatile double * __restrict minyd, volatile double * __restrict minzd)
{
  register int i, j, k, inc;
  register double val, minx, maxx, miny, maxy, minz, maxz;
  __shared__ volatile double sminx[THREADS1], smaxx[THREADS1], sminy[THREADS1], smaxy[THREADS1], sminz[THREADS1], smaxz[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  minx = maxx = posxd[0];
  miny = maxy = posyd[0];
  minz = maxz = poszd[0];

  // scan all bodies
  i = threadIdx.x;
  inc = THREADS1 * gridDim.x;
  for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc) {
    val = posxd[j];
    minx = fminf(minx, val);
    maxx = fmaxf(maxx, val);
    val = posyd[j];
    miny = fminf(miny, val);
    maxy = fmaxf(maxy, val);
    val = poszd[j];
    minz = fminf(minz, val);
    maxz = fmaxf(maxz, val);
  }

  // reduction in shared memory
  sminx[i] = minx;
  smaxx[i] = maxx;
  sminy[i] = miny;
  smaxy[i] = maxy;
  sminz[i] = minz;
  smaxz[i] = maxz;

  for (j = THREADS1 / 2; j > 0; j /= 2) {
    __syncthreads();
    if (i < j) {
      k = i + j;
      sminx[i] = minx = fminf(minx, sminx[k]);
      smaxx[i] = maxx = fmaxf(maxx, smaxx[k]);
      sminy[i] = miny = fminf(miny, sminy[k]);
      smaxy[i] = maxy = fmaxf(maxy, smaxy[k]);
      sminz[i] = minz = fminf(minz, sminz[k]);
      smaxz[i] = maxz = fmaxf(maxz, smaxz[k]);
    }
  }

  // write block result to global memory
  if (i == 0) {
    k = blockIdx.x;
    minxd[k] = minx;
    maxxd[k] = maxx;
    minyd[k] = miny;
    maxyd[k] = maxy;
    minzd[k] = minz;
    maxzd[k] = maxz;
    __threadfence();

    inc = gridDim.x - 1;
    if (inc == atomicInc(&blkcntd, inc)) {
      // I'm the last block, so combine all block results
      for (j = 0; j <= inc; j++) {
        minx = fminf(minx, minxd[j]);
        maxx = fmaxf(maxx, maxxd[j]);
        miny = fminf(miny, minyd[j]);
        maxy = fmaxf(maxy, maxyd[j]);
        minz = fminf(minz, minzd[j]);
        maxz = fmaxf(maxz, maxzd[j]);
      }

      // compute 'radius'
      val = fmaxf(maxx - minx, maxy - miny);
      radiusd = fmaxf(val, maxz - minz) * 0.5f;

      // create root node
      k = nnodesd;
      bottomd = k;

      massd[k] = -1.0f;
      startd[k] = 0;
      posxd[k] = (minx + maxx) * 0.5f;
      posyd[k] = (miny + maxy) * 0.5f;
      poszd[k] = (minz + maxz) * 0.5f;
      k *= 8;
      for (i = 0; i < 8; i++) childd[k + i] = -1;

      stepd++;
    }
  }
}


/******************************************************************************/
/*** build tree ***************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(1024, 1)
void ClearKernel1(int nnodesd, int nbodiesd, volatile int * __restrict childd)
{
  register int k, inc, top, bottom;

  top = 8 * nnodesd;
  bottom = 8 * nbodiesd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < top) {
    childd[k] = -1;
    k += inc;
  }
}


__global__
__launch_bounds__(THREADS2, FACTOR2)
void TreeBuildingKernel(int nnodesd, int nbodiesd, volatile int * __restrict errd, volatile int * __restrict childd, volatile double * __restrict posxd, volatile double * __restrict posyd, volatile double * __restrict poszd)
{
  register int i, j, depth, localmaxdepth, skip, inc;
  register double x, y, z, r;
  register double px, py, pz;
  register double dx, dy, dz;
  register int ch, n, cell, locked, patch;
  register double radius, rootx, rooty, rootz;

  // cache root data
  radius = radiusd;
  rootx = posxd[nnodesd];
  rooty = posyd[nnodesd];
  rootz = poszd[nnodesd];

  localmaxdepth = 1;
  skip = 1;
  inc = blockDim.x * gridDim.x;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < nbodiesd) {
    if (skip != 0) {
      // new body, so start traversing at root
      skip = 0;
      px = posxd[i];
      py = posyd[i];
      pz = poszd[i];
      n = nnodesd;
      depth = 1;
      r = radius * 0.5f;
      dx = dy = dz = -r;
      j = 0;
      // determine which child to follow
      if (rootx < px) {j = 1; dx = r;}
      if (rooty < py) {j |= 2; dy = r;}
      if (rootz < pz) {j |= 4; dz = r;}
      x = rootx + dx;
      y = rooty + dy;
      z = rootz + dz;
    }

    // follow path to leaf cell
    ch = childd[n*8+j];
    while (ch >= nbodiesd) {
      n = ch;
      depth++;
      r *= 0.5f;
      dx = dy = dz = -r;
      j = 0;
      // determine which child to follow
      if (x < px) {j = 1; dx = r;}
      if (y < py) {j |= 2; dy = r;}
      if (z < pz) {j |= 4; dz = r;}
      x += dx;
      y += dy;
      z += dz;
      ch = childd[n*8+j];
    }

    if (ch != -2) {  // skip if child pointer is locked and try again later
      locked = n*8+j;
      if (ch == -1) {
        if (-1 == atomicCAS((int *)&childd[locked], -1, i)) {  // if null, just insert the new body
          localmaxdepth = max(depth, localmaxdepth);
          i += inc;  // move on to next body
          skip = 1;
        }
      } else {  // there already is a body in this position
        if (ch == atomicCAS((int *)&childd[locked], ch, -2)) {  // try to lock
          patch = -1;
          // create new cell(s) and insert the old and new body
          do {
            depth++;

            cell = atomicSub((int *)&bottomd, 1) - 1;
            if (cell <= nbodiesd) {
              *errd = 1;
              bottomd = nnodesd;
            }

            if (patch != -1) {
              childd[n*8+j] = cell;
            }
            patch = max(patch, cell);

            j = 0;
            if (x < posxd[ch]) j = 1;
            if (y < posyd[ch]) j |= 2;
            if (z < poszd[ch]) j |= 4;
            childd[cell*8+j] = ch;

            n = cell;
            r *= 0.5f;
            dx = dy = dz = -r;
            j = 0;
            if (x < px) {j = 1; dx = r;}
            if (y < py) {j |= 2; dy = r;}
            if (z < pz) {j |= 4; dz = r;}
            x += dx;
            y += dy;
            z += dz;

            ch = childd[n*8+j];
            // repeat until the two bodies are different children
          } while (ch >= 0);
          childd[n*8+j] = i;

          localmaxdepth = max(depth, localmaxdepth);
          i += inc;  // move on to next body
          skip = 2;
        }
      }
    }
    __syncthreads();  // __threadfence();

    if (skip == 2) {
      childd[locked] = patch;
    }
  }
  // record maximum tree depth
  atomicMax((int *)&maxdepthd, localmaxdepth);
}


__global__
__launch_bounds__(1024, 1)
void ClearKernel2(int nnodesd, volatile int * __restrict startd, volatile double * __restrict massd)
{
  register int k, inc, bottom;

  bottom = bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < nnodesd) {
    massd[k] = -1.0f;
    startd[k] = -1;
    k += inc;
  }
}


/******************************************************************************/
/*** compute center of mass ***************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS3, FACTOR3)
void SummarizationKernel(const int nnodesd, const int nbodiesd, volatile int * __restrict countd, const int * __restrict childd, volatile double * __restrict massd, volatile double * __restrict posxd, volatile double * __restrict posyd, volatile double * __restrict poszd)
{
  register int i, j, k, ch, inc, cnt, bottom, flag;
  register double m, cm, px, py, pz;
  __shared__ int child[THREADS3 * 8];
  __shared__ double mass[THREADS3 * 8];

  bottom = bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  register int restart = k;
  for (j = 0; j < 5; j++) {  // wait-free pre-passes
    // iterate over all cells assigned to thread
    while (k <= nnodesd) {
      if (massd[k] < 0.0f) {
        for (i = 0; i < 8; i++) {
          ch = childd[k*8+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch >= nbodiesd) && ((mass[i*THREADS3+threadIdx.x] = massd[ch]) < 0.0f)) {
            break;
          }
        }
        if (i == 8) {
          // all children are ready
          cm = 0.0f;
          px = 0.0f;
          py = 0.0f;
          pz = 0.0f;
          cnt = 0;
          for (i = 0; i < 8; i++) {
            ch = child[i*THREADS3+threadIdx.x];
            if (ch >= 0) {
              if (ch >= nbodiesd) {  // count bodies (needed later)
                m = mass[i*THREADS3+threadIdx.x];
                cnt += countd[ch];
              } else {
                m = massd[ch];
                cnt++;
              }
              // add child's contribution
              cm += m;
              px += posxd[ch] * m;
              py += posyd[ch] * m;
              pz += poszd[ch] * m;
            }
          }
          countd[k] = cnt;
          m = 1.0f / cm;
          posxd[k] = px * m;
          posyd[k] = py * m;
          poszd[k] = pz * m;
          __threadfence();  // make sure data are visible before setting mass
          massd[k] = cm;
        }
      }
      k += inc;  // move on to next cell
    }
    k = restart;
  }

  flag = 0;
  j = 0;
  // iterate over all cells assigned to thread
  while (k <= nnodesd) {
    if (massd[k] >= 0.0f) {
      k += inc;
    } else {
      if (j == 0) {
        j = 8;
        for (i = 0; i < 8; i++) {
          ch = childd[k*8+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch < nbodiesd) || ((mass[i*THREADS3+threadIdx.x] = massd[ch]) >= 0.0f)) {
            j--;
          }
        }
      } else {
        j = 8;
        for (i = 0; i < 8; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if ((ch < nbodiesd) || (mass[i*THREADS3+threadIdx.x] >= 0.0f) || ((mass[i*THREADS3+threadIdx.x] = massd[ch]) >= 0.0f)) {
            j--;
          }
        }
      }

      if (j == 0) {
        // all children are ready
        cm = 0.0f;
        px = 0.0f;
        py = 0.0f;
        pz = 0.0f;
        cnt = 0;
        for (i = 0; i < 8; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if (ch >= 0) {
            if (ch >= nbodiesd) {  // count bodies (needed later)
              m = mass[i*THREADS3+threadIdx.x];
              cnt += countd[ch];
            } else {
              m = massd[ch];
              cnt++;
            }
            // add child's contribution
            cm += m;
            px += posxd[ch] * m;
            py += posyd[ch] * m;
            pz += poszd[ch] * m;
          }
        }
        countd[k] = cnt;
        m = 1.0f / cm;
        posxd[k] = px * m;
        posyd[k] = py * m;
        poszd[k] = pz * m;
        flag = 1;
      }
    }
    __syncthreads();  // __threadfence();
    if (flag != 0) {
      massd[k] = cm;
      k += inc;
      flag = 0;
    }
  }
}


/******************************************************************************/
/*** sort bodies **************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS4, FACTOR4)
void SortKernel(int nnodesd, int nbodiesd, int * __restrict sortd, int * __restrict countd, volatile int * __restrict startd, int * __restrict childd)
{
  register int i, j, k, ch, dec, start, bottom;

  bottom = bottomd;
  dec = blockDim.x * gridDim.x;
  k = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all cells assigned to thread
  while (k >= bottom) {
    start = startd[k];
    if (start >= 0) {
      j = 0;
      for (i = 0; i < 8; i++) {
        ch = childd[k*8+i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front (needed later for speed)
            childd[k*8+i] = -1;
            childd[k*8+j] = ch;
          }
          j++;
          if (ch >= nbodiesd) {
            // child is a cell
            startd[ch] = start;  // set start ID of child
            start += countd[ch];  // add #bodies in subtree
          } else {
            // child is a body
            sortd[start] = ch;  // record body in 'sorted' array
            start++;
          }
        }
      }
      k -= dec;  // move on to next cell
    }
  }
}


/******************************************************************************/
/*** compute force ************************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS5, FACTOR5)
void ForceCalculationKernel(int nnodesd, int startnbodiesd, int endnbodiesd, int nbodiesd, volatile int * __restrict errd, double dthfd, double itolsqd, double epssqd, volatile int * __restrict sortd, volatile int * __restrict childd, volatile double * __restrict massd, volatile double * __restrict posxd, volatile double * __restrict posyd, volatile double * __restrict poszd, volatile double * __restrict velxd, volatile double * __restrict velyd, volatile double * __restrict velzd, volatile double * __restrict accxd, volatile double * __restrict accyd, volatile double * __restrict acczd)
{

  register int i, j, k, n, depth, base, sbase, diff, pd, nd;
  register double px, py, pz, ax, ay, az, dx, dy, dz, tmp;
  __shared__ volatile int pos[MAXDEPTH * THREADS5/WARPSIZE], node[MAXDEPTH * THREADS5/WARPSIZE];
  __shared__ double dq[MAXDEPTH * THREADS5/WARPSIZE];

  if (0 == threadIdx.x) {

    tmp = radiusd * 2;
    // precompute values that depend only on tree level
    dq[0] = tmp * tmp * itolsqd;
    for (i = 1; i < maxdepthd; i++) {
      dq[i] = dq[i - 1] * 0.25f;
      dq[i - 1] += epssqd;
    }
    dq[i - 1] += epssqd;

    if (maxdepthd > MAXDEPTH) {
      *errd = maxdepthd;
    }
  }
  __syncthreads();

  if (maxdepthd <= MAXDEPTH) {
    // figure out first thread in each warp (lane 0)
    base = threadIdx.x / WARPSIZE;
    sbase = base * WARPSIZE;
    j = base * MAXDEPTH;

    diff = threadIdx.x - sbase;
    // make multiple copies to avoid index calculations later
    if (diff < MAXDEPTH) {
      dq[diff+j] = dq[diff];
    }
    __syncthreads();
    __threadfence_block();

    // iterate over all bodies assigned to thread
    for (k = startnbodiesd + threadIdx.x + blockIdx.x * blockDim.x; k < endnbodiesd; k += blockDim.x * gridDim.x) {
      i = sortd[k];  // get permuted/sorted index
      // cache position info
      px = posxd[i];
      py = posyd[i];
      pz = poszd[i];

      ax = 0.0f;
      ay = 0.0f;
      az = 0.0f;

      // initialize iteration stack, i.e., push root node onto stack
      depth = j;
      if (sbase == threadIdx.x) {
        pos[j] = 0;
        node[j] = nnodesd * 8;
      }

      do {
        // stack is not empty
        pd = pos[depth];
        nd = node[depth];
        while (pd < 8) {
          // node on top of stack has more children to process
          n = childd[nd + pd];  // load child pointer
          pd++;

          if (n >= 0) {
            dx = posxd[n] - px;
            dy = posyd[n] - py;
            dz = poszd[n] - pz;
            tmp = dx*dx + (dy*dy + (dz*dz + epssqd));  // compute distance squared (plus softening)
            if ((n < nbodiesd) || __all(tmp >= dq[depth])) {  // check if all threads agree that cell is far enough away (or is a body)
              tmp = rsqrtf(tmp);  // compute distance
              tmp = massd[n] * tmp * tmp * tmp;
              ax += dx * tmp;
              ay += dy * tmp;
              az += dz * tmp;
            } else {
              // push cell onto stack
              if (sbase == threadIdx.x) {  // maybe don't push and inc if last child
                pos[depth] = pd;
                node[depth] = nd;
              }
              depth++;
              pd = 0;
              nd = n * 8;
            }
          } else {
            pd = 8;  // early out because all remaining children are also zero
          }
        }
        depth--;  // done with this level
      } while (depth >= j);

      if (stepd > 0) {
        // update velocity
        velxd[i] += (ax - accxd[i]) * dthfd;
        velyd[i] += (ay - accyd[i]) * dthfd;
        velzd[i] += (az - acczd[i]) * dthfd;
      }

      // save computed acceleration
      accxd[i] = ax;
      accyd[i] = ay;
      acczd[i] = az;
    }
  }
}

/******************************************************************************/
/*** Reduce results ***********************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS7, FACTOR7)
void TransferKernel(int startnbodiesd, int endnbodiesd, volatile int * __restrict sortd, volatile double * __restrict accxd, volatile double * __restrict accyd, volatile double * __restrict acczd, volatile double * __restrict temp_accxd, volatile double * __restrict temp_accyd, volatile double * __restrict temp_acczd){
  int tid, i;

  for(tid = startnbodiesd + threadIdx.x + blockIdx.x * blockDim.x; tid < endnbodiesd; tid += blockDim.x * gridDim.x){
    i = sortd[tid];

    accxd[i] = temp_accxd[i];
    accyd[i] = temp_accyd[i];
    acczd[i] = temp_acczd[i];

  }
}

/******************************************************************************/
/*** advance bodies ***********************************************************/
/******************************************************************************/

__global__
__launch_bounds__(THREADS6, FACTOR6)
void IntegrationKernel(int nbodiesd, double dtimed, double dthfd, volatile double * __restrict posxd, volatile double * __restrict posyd, volatile double * __restrict poszd, volatile double * __restrict velxd, volatile double * __restrict velyd, volatile double * __restrict velzd, volatile double * __restrict accxd, volatile double * __restrict accyd, volatile double * __restrict acczd)
{
  register int i, inc;
  register double dvelx, dvely, dvelz;
  register double velhx, velhy, velhz;

  // iterate over all bodies assigned to thread
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += inc) {
    // integrate
    dvelx = accxd[i] * dthfd;
    dvely = accyd[i] * dthfd;
    dvelz = acczd[i] * dthfd;

    velhx = velxd[i] + dvelx;
    velhy = velyd[i] + dvely;
    velhz = velzd[i] + dvelz;

    posxd[i] += velhx * dtimed;
    posyd[i] += velhy * dtimed;
    poszd[i] += velhz * dtimed;

    velxd[i] = velhx + dvelx;
    velyd[i] = velhy + dvely;
    velzd[i] = velhz + dvelz;
  }
}


/******************************************************************************/

static void CudaTest(const char *msg)
{
  cudaError_t e;

  cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}


/******************************************************************************/

// random number generator

#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;


static void drndset(int seed)
{
   A = 1;
   B = 0;
   randx = (A * seed + B) & MASK;
   A = (MULT * A) & MASK;
   B = (MULT * B + ADD) & MASK;
}


static double drnd()
{
   lastrand = randx;
   randx = (A * randx + B) & MASK;
   return (double)lastrand / TWOTO31;
}


/******************************************************************************/


int main(int argc, char *argv[])
{

  register int i, mainBlocks, secondaryBlocks, mainBodiesBegin, secondaryBodiesBegin, mainBodiesEnd, secondaryBodiesEnd;
  int nnodes, nbodies, step, timesteps;
  //register double runtime;
  //register int run;
  register double dtime, dthf, epssq, itolsq;
  //double time, timing[7];
  //cudaEvent_t start, stop;
  int main_error = 0, secondary_error = 0, error= 0;
  double *mass, *posx, *posy, *posz, *velx, *vely, *velz;

  int *main_errl, *main_sortl, *main_childl, *main_countl, *main_startl;
  double *main_massl;
  double *main_posxl, *main_posyl, *main_poszl;
  double *main_velxl, *main_velyl, *main_velzl;
  double *main_accxl, *main_accyl, *main_acczl;
  double *main_temp_accxl, *main_temp_accyl, *main_temp_acczl;

  int *secondary_errl, *secondary_sortl, *secondary_childl, *secondary_countl, *secondary_startl;
  double *secondary_massl;
  double *secondary_posxl, *secondary_posyl, *secondary_poszl;
  double *secondary_velxl, *secondary_velyl, *secondary_velzl;
  double *secondary_accxl, *secondary_accyl, *secondary_acczl;
  
  double *maxxl, *maxyl, *maxzl;
  double *minxl, *minyl, *minzl;
  register double rsc, vsc, r, v, x, y, z, sq, scale;

  cudaStream_t mainTransferStream, secondaryTransferStream; 

  FILE *output;
  output = fopen("output.txt", "w");

  // perform some checks

  printf("CUDA BarnesHut v3.1 ");
  #ifdef __KEPLER__
   printf("[Kepler]\n");
  #else
   printf("[Fermi]\n");
  #endif
    printf("Copyright (c) 2013, Texas State University-San Marcos. All rights reserved.\n");
    fflush(stdout);
  if (argc != 5) {
    fprintf(stderr, "\n");
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps main_device auxiliary_device\n");
    exit(-1);
  }

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "There is no device supporting CUDA\n");
    exit(-1);
  }

  const int mainDevice = atoi(argv[3]);
  if ((mainDevice < 0) || (deviceCount <= mainDevice)) {
    fprintf(stderr, "There is no device %d\n", mainDevice);
    exit(-1);
  }

  const int secondaryDevice = atoi(argv[4]);
  if ((secondaryDevice < 0) || (deviceCount <= secondaryDevice)) {
    fprintf(stderr, "There is no device %d\n", secondaryDevice);
    exit(-1);
  }

  if(secondaryDevice == mainDevice){
    fprintf(stderr, "Secondary device must different from main device\n");
    exit(-1);
  }

  cudaDeviceProp deviceProp;

//-------------------------------------------------------------------------------------------------------
//Device properties and function configurations for main device

  cudaSetDevice(mainDevice);

  cudaStreamCreate(&mainTransferStream);

  cudaGetDeviceProperties(&deviceProp, mainDevice);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
   fprintf(stderr, "Device %d is not CUDA capable\n", mainDevice);
   exit(-1);
  }

  printf("Main Device: %s\n", deviceProp.name);

  if (deviceProp.major < 2) {
    fprintf(stderr, "Device %d needs at least compute capability 2.0\n", mainDevice);
    exit(-1);
  }

  if (deviceProp.warpSize != WARPSIZE) {
    fprintf(stderr, "Device %d. Warp size must be %d\n", mainDevice, deviceProp.warpSize);
   exit(-1);
  }

  mainBlocks = deviceProp.multiProcessorCount;

    // set L1/shared memory configuration
  cudaFuncSetCacheConfig(BoundingBoxKernel  , cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TreeBuildingKernel , cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel1       , cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel2       , cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(SortKernel         , cudaFuncCachePreferL1);
  #ifdef __KEPLER__
    cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferEqual);
  #else
    cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
  #endif
    cudaFuncSetCacheConfig(IntegrationKernel     , cudaFuncCachePreferL1);

  cudaGetLastError();  // reset error value

//-------------------------------------------------------------------------------------------------------
//Device properties and function configurations for secondary device

  cudaSetDevice(secondaryDevice);
  cudaStreamCreate(&secondaryTransferStream);

  cudaGetDeviceProperties(&deviceProp, secondaryDevice);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
   fprintf(stderr, "Device %d is not CUDA capable\n", secondaryDevice);
   exit(-1);
  }

  printf("Auxiliary Device: %s\n", deviceProp.name);

  if (deviceProp.major < 2) {
    fprintf(stderr, "Device %d needs at least compute capability 2.0\n", secondaryDevice);
    exit(-1);
  }

  if (deviceProp.warpSize != WARPSIZE) {
    fprintf(stderr, "Device %d. Warp size must be %d\n", secondaryDevice, deviceProp.warpSize);
   exit(-1);
  }

  secondaryBlocks = deviceProp.multiProcessorCount;

    // set L1/shared memory configuration
  cudaFuncSetCacheConfig(BoundingBoxKernel  , cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TreeBuildingKernel , cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel1       , cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(ClearKernel2       , cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(SortKernel         , cudaFuncCachePreferL1);
  #ifdef __KEPLER__
    cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferEqual);
  #else
    cudaFuncSetCacheConfig(ForceCalculationKernel, cudaFuncCachePreferL1);
  #endif
    cudaFuncSetCacheConfig(IntegrationKernel     , cudaFuncCachePreferL1);

  cudaGetLastError();  // reset error value

//-------------------------------------------------------------------------------------------------------

  if ((WARPSIZE <= 0) || (WARPSIZE & (WARPSIZE-1) != 0)) {
    fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
    exit(-1);
  }
  if (MAXDEPTH > WARPSIZE) {
    fprintf(stderr, "MAXDEPTH must be less than or equal to WARPSIZE\n");
    exit(-1);
  }
  if ((THREADS1 <= 0) || (THREADS1 & (THREADS1-1) != 0)) {
    fprintf(stderr, "THREADS1 must be greater than zero and a power of two\n");
    exit(-1);
  }

  nbodies = atoi(argv[1]);
  if (nbodies < 1) {
    fprintf(stderr, "nbodies is too small: %d\n", nbodies);
    exit(-1);
  }

  if (nbodies > (1 << 30)) {
    fprintf(stderr, "nbodies is too large: %d\n", nbodies);
    exit(-1);
  }

  mainBodiesBegin      = 0;
  mainBodiesEnd        = 0.5 * nbodies;
  secondaryBodiesBegin = mainBodiesEnd;
  secondaryBodiesEnd   = nbodies;

  nnodes = nbodies * 2;
  if (nnodes < 1024*mainBlocks) nnodes = 1024*mainBlocks;

  while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
  nnodes--;

  timesteps = atoi(argv[2]);
  dtime = 0.025;  dthf = dtime * 0.5f;
  epssq = 0.05 * 0.05;
  itolsq = 1.0f / (0.5 * 0.5);

  printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

  //Allocate host memory for bodies information
  if (cudaSuccess != cudaMallocHost((void **)&mass,  sizeof(double) * nbodies))     fprintf(stderr, "could not allocate host mass\n");  //CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMallocHost((void **)&posx,  sizeof(double) * nbodies))     fprintf(stderr, "could not allocate host posx\n");  //CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMallocHost((void **)&posy,  sizeof(double) * nbodies))     fprintf(stderr, "could not allocate host posy\n");  //CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMallocHost((void **)&posz,  sizeof(double) * nbodies))     fprintf(stderr, "could not allocate host posz\n");  //CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMallocHost((void **)&velx,  sizeof(double) * nbodies))     fprintf(stderr, "could not allocate host velx\n");  //CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMallocHost((void **)&vely,  sizeof(double) * nbodies))     fprintf(stderr, "could not allocate host vely\n");  //CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMallocHost((void **)&velz,  sizeof(double) * nbodies))     fprintf(stderr, "could not allocate host velz\n");  //CudaTest("couldn't allocate massd");


  cudaSetDevice(mainDevice);

  if (cudaSuccess != cudaMalloc((void **)&main_errl,        sizeof(int)))                           fprintf(stderr, "could not allocate errd on main device\n");   //CudaTest("couldn't allocate errd");
  if (cudaSuccess != cudaMalloc((void **)&main_childl,      sizeof(int)    * (nnodes+1) * 8))       fprintf(stderr, "could not allocate childd on main device\n"); //CudaTest("couldn't allocate childd");
  if (cudaSuccess != cudaMalloc((void **)&main_massl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate massd on main device\n");  //CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMalloc((void **)&main_posxl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate posxd on main device\n");  //CudaTest("couldn't allocate posxd");
  if (cudaSuccess != cudaMalloc((void **)&main_posyl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate posyd on main device\n");  //CudaTest("couldn't allocate posyd");
  if (cudaSuccess != cudaMalloc((void **)&main_poszl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate poszd on main device\n");  //CudaTest("couldn't allocate poszd");
  if (cudaSuccess != cudaMalloc((void **)&main_velxl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate velxd on main device\n");  //CudaTest("couldn't allocate velxd");
  if (cudaSuccess != cudaMalloc((void **)&main_velyl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate velyd on main device\n");  //CudaTest("couldn't allocate velyd");
  if (cudaSuccess != cudaMalloc((void **)&main_velzl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate velzd on main device\n");  //CudaTest("couldn't allocate velzd");
  if (cudaSuccess != cudaMalloc((void **)&main_accxl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate accxd on main device\n");  //CudaTest("couldn't allocate accxd");
  if (cudaSuccess != cudaMalloc((void **)&main_accyl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate accyd on main device\n");  //CudaTest("couldn't allocate accyd");
  if (cudaSuccess != cudaMalloc((void **)&main_acczl,       sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate acczd on main device\n");  //CudaTest("couldn't allocate acczd");
  if (cudaSuccess != cudaMalloc((void **)&main_countl,      sizeof(int)    * (nnodes+1)))           fprintf(stderr, "could not allocate countd on main device\n"); //CudaTest("couldn't allocate countd");
  if (cudaSuccess != cudaMalloc((void **)&main_startl,      sizeof(int)    * (nnodes+1)))           fprintf(stderr, "could not allocate startd on main device\n"); //CudaTest("couldn't allocate startd");
  if (cudaSuccess != cudaMalloc((void **)&main_sortl,       sizeof(int)    * (nnodes+1)))           fprintf(stderr, "could not allocate sortd on main device\n");  //CudaTest("couldn't allocate sortd");

  if (cudaSuccess != cudaMalloc((void **)&main_temp_accxl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate temp_accxd on main device\n");  //CudaTest("couldn't allocate temp_accxd");
  if (cudaSuccess != cudaMalloc((void **)&main_temp_accyl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate temp_accyd on main device\n");  //CudaTest("couldn't allocate temp_accyd");
  if (cudaSuccess != cudaMalloc((void **)&main_temp_acczl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate temp_acczd on main device\n");  //CudaTest("couldn't allocate temp_acczd");

  if (cudaSuccess != cudaMalloc((void **)&maxxl,            sizeof(double) * mainBlocks * FACTOR1)) fprintf(stderr, "could not allocate maxxd on main device\n");  //CudaTest("couldn't allocate maxxd");
  if (cudaSuccess != cudaMalloc((void **)&maxyl,            sizeof(double) * mainBlocks * FACTOR1)) fprintf(stderr, "could not allocate maxyd on main device\n");  //CudaTest("couldn't allocate maxyd");
  if (cudaSuccess != cudaMalloc((void **)&maxzl,            sizeof(double) * mainBlocks * FACTOR1)) fprintf(stderr, "could not allocate maxzd on main device\n");  //CudaTest("couldn't allocate maxzd");
  if (cudaSuccess != cudaMalloc((void **)&minxl,            sizeof(double) * mainBlocks * FACTOR1)) fprintf(stderr, "could not allocate minxd on main device\n");  //CudaTest("couldn't allocate minxd");
  if (cudaSuccess != cudaMalloc((void **)&minyl,            sizeof(double) * mainBlocks * FACTOR1)) fprintf(stderr, "could not allocate minyd on main device\n");  //CudaTest("couldn't allocate minyd");
  if (cudaSuccess != cudaMalloc((void **)&minzl,            sizeof(double) * mainBlocks * FACTOR1)) fprintf(stderr, "could not allocate minzd on main device\n");  //CudaTest("couldn't allocate minzd");

  cudaSetDevice(secondaryDevice);

  if (cudaSuccess != cudaMalloc((void **)&secondary_errl,   sizeof(int)))                           fprintf(stderr, "could not allocate errd on secondary device\n");   //CudaTest("couldn't allocate errd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_childl, sizeof(int)    * (nnodes+1) * 8))       fprintf(stderr, "could not allocate childd on secondary device\n"); //CudaTest("couldn't allocate childd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_massl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate massd on secondary device\n");  //CudaTest("couldn't allocate massd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_posxl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate posxd on secondary device\n");  //CudaTest("couldn't allocate posxd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_posyl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate posyd on secondary device\n");  //CudaTest("couldn't allocate posyd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_poszl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate poszd on secondary device\n");  //CudaTest("couldn't allocate poszd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_velxl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate velxd on secondary device\n");  //CudaTest("couldn't allocate velxd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_velyl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate velyd on secondary device\n");  //CudaTest("couldn't allocate velyd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_velzl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate velzd on secondary device\n");  //CudaTest("couldn't allocate velzd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_accxl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate accxd on secondary device\n");  //CudaTest("couldn't allocate accxd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_accyl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate accyd on secondary device\n");  //CudaTest("couldn't allocate accyd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_acczl,  sizeof(double) * (nnodes+1)))           fprintf(stderr, "could not allocate acczd on secondary device\n");  //CudaTest("couldn't allocate acczd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_countl, sizeof(int)    * (nnodes+1)))           fprintf(stderr, "could not allocate countd on secondary device\n"); //CudaTest("couldn't allocate countd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_startl, sizeof(int)    * (nnodes+1)))           fprintf(stderr, "could not allocate startd on secondary device\n"); //CudaTest("couldn't allocate startd");
  if (cudaSuccess != cudaMalloc((void **)&secondary_sortl,  sizeof(int)    * (nnodes+1)))           fprintf(stderr, "could not allocate sortd on secondary device\n");  //CudaTest("couldn't allocate sortd");

  int access;

  cudaDeviceCanAccessPeer(&access, mainDevice, secondaryDevice);

  if(access){
    cudaDeviceEnablePeerAccess(mainDevice, 0);
  }

  cudaSetDevice(mainDevice);

  if(access){
    cudaDeviceEnablePeerAccess(secondaryDevice, 0);
  }
  

  // generate input

  drndset(7);
  rsc = (3 * 3.1415926535897932384626433832795) / 16;
  vsc = sqrt(1.0 / rsc);
  for (i = 0; i < nbodies; i++) {
    mass[i] = 1.0 / nbodies;
    r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);
    posx[i] = x * scale;
    posy[i] = y * scale;
    posz[i] = z * scale;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x*x * pow(1 - x*x, 3.5));
    v = x * sqrt(2.0 / sqrt(1 + r*r));
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    velx[i] = x * scale;
    vely[i] = y * scale;
    velz[i] = z * scale;
  }


  for(i = 0; i < nbodies; i++){
    fprintf(output, "%f "  , posx[i]);
    fprintf(output, "%f "  , posy[i]);
    fprintf(output, "%f \n", posz[i]);
  }

  if (cudaSuccess != cudaMemcpy(main_massl, mass, sizeof(double) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of mass to main device failed\n");  //CudaTest("mass copy to device failed");
  if (cudaSuccess != cudaMemcpy(main_posxl, posx, sizeof(double) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posx to main device failed\n");  //CudaTest("posx copy to device failed");
  if (cudaSuccess != cudaMemcpy(main_posyl, posy, sizeof(double) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posy to main device failed\n");  //CudaTest("posy copy to device failed");
  if (cudaSuccess != cudaMemcpy(main_poszl, posz, sizeof(double) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of posz to main device failed\n");  //CudaTest("posz copy to device failed");
  if (cudaSuccess != cudaMemcpy(main_velxl, velx, sizeof(double) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of velx to main device failed\n");  //CudaTest("velx copy to device failed");
  if (cudaSuccess != cudaMemcpy(main_velyl, vely, sizeof(double) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of vely to main device failed\n");  //CudaTest("vely copy to device failed");
  if (cudaSuccess != cudaMemcpy(main_velzl, velz, sizeof(double) * nbodies, cudaMemcpyHostToDevice)) fprintf(stderr, "copying of velz to main device failed\n");  //CudaTest("velz copy to device failed");

  //Free unnecessary arrays
  cudaFreeHost(mass);
  cudaFreeHost(velx);
  cudaFreeHost(vely);
  cudaFreeHost(velz);

  clock_t begin = clock();

  InitializationKernel<<<1, 1>>>(main_errl);
  cudaDeviceSynchronize();
  CudaTest("InitializationKernel launch failed");


  for (step = 0; step < timesteps; step++) {
  
    BoundingBoxKernel<<<mainBlocks * FACTOR1, THREADS1, 0>>>(nnodes, nbodies, main_startl, main_childl, main_massl, main_posxl, main_posyl, main_poszl, maxxl, maxyl, maxzl, minxl, minyl, minzl);
    CudaTest("BoundingBoxKernel launch failed");
      
    ClearKernel1<<<mainBlocks * 1, 1024, 0>>>(nnodes, nbodies, main_childl);
    TreeBuildingKernel<<<mainBlocks * FACTOR2, THREADS2, 0>>>(nnodes, nbodies, main_errl, main_childl, main_posxl, main_posyl, main_poszl);
    ClearKernel2<<<mainBlocks * 1, 1024, 0>>>(nnodes, main_startl, main_massl);
    CudaTest("TreeBuildingKernel launch failed");

    SummarizationKernel<<<mainBlocks * FACTOR3, THREADS3, 0>>>(nnodes, nbodies, main_countl, main_childl, main_massl, main_posxl, main_posyl, main_poszl);
    CudaTest("SummarizationKernel launch failed");

    SortKernel<<<mainBlocks * FACTOR4, THREADS4, 0>>>(nnodes, nbodies, main_sortl, main_countl, main_startl, main_childl);
    CudaTest("SortKernel launch failed");


    cudaSetDevice(mainDevice);

    cudaMemcpyFromSymbol(&h_stepd    , stepd    , sizeof(int)   , 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&h_maxdepthd, maxdepthd, sizeof(int)   , 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&h_radiusd  , radiusd  , sizeof(double), 0, cudaMemcpyDeviceToHost);

    cudaSetDevice(secondaryDevice);

    cudaMemcpyToSymbol(stepd    , &h_stepd    , sizeof(int)   , 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(maxdepthd, &h_maxdepthd, sizeof(int)   , 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(radiusd  , &h_radiusd  , sizeof(double), 0, cudaMemcpyHostToDevice);

    //Transfer bodies information from main device to secondary device
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_errl,   secondaryDevice, main_errl,   mainDevice,  sizeof(int)                    , secondaryTransferStream)) fprintf(stderr, "copying of err from main device to secondary device failed\n");     //CudaTest("err copy to device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_sortl,  secondaryDevice, main_sortl,  mainDevice,  sizeof(int)    * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of sort from main device to secondary device failed\n");    //CudaTest("sort copy to device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_childl, secondaryDevice, main_childl, mainDevice,  sizeof(int)    * (nnodes+1) * 8, secondaryTransferStream)) fprintf(stderr, "copying of child from main device to secondary device failed\n");   //CudaTest("child copy to device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_massl,  secondaryDevice, main_massl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of mass from main device to secondary device failed\n");    //CudaTest("mass copy to device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_posxl,  secondaryDevice, main_posxl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of posx from main device to secondary device failed\n");  //CudaTest("posx copy from device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_posyl,  secondaryDevice, main_posyl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of posy from main device to secondary device failed\n");  //CudaTest("posy copy from device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_poszl,  secondaryDevice, main_poszl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of posz from main device to secondary device failed\n");  //CudaTest("posz copy from device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_velxl,  secondaryDevice, main_velxl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of velx from main device to secondary device failed\n");  //CudaTest("velx copy from device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_velyl,  secondaryDevice, main_velyl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of vely from main device to secondary device failed\n");  //CudaTest("vely copy from device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_velzl,  secondaryDevice, main_velzl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of velz from main device to secondary device failed\n");  //CudaTest("velz copy from device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_accxl,  secondaryDevice, main_accxl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of accx from main device to secondary device failed\n");  //CudaTest("accx copy from device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_accyl,  secondaryDevice, main_accyl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of accy from main device to secondary device failed\n");  //CudaTest("accy copy from device failed");
    if (cudaSuccess != cudaMemcpyPeerAsync(secondary_acczl,  secondaryDevice, main_acczl,  mainDevice,  sizeof(double) * (nnodes+1)    , secondaryTransferStream)) fprintf(stderr, "copying of accz from main device to secondary device failed\n");  //CudaTest("accz copy from device failed");


    //Each GPU then calculates their half of the interactions
    cudaSetDevice(mainDevice);

    ForceCalculationKernel<<<mainBlocks * FACTOR5, THREADS5, 0>>>(nnodes, mainBodiesBegin, mainBodiesEnd, nbodies, main_errl, dthf, itolsq, epssq, main_sortl, main_childl, main_massl, main_posxl, main_posyl, main_poszl, main_velxl, main_velyl, main_velzl, main_accxl, main_accyl, main_acczl);
        

    cudaSetDevice(secondaryDevice);
    cudaStreamSynchronize(secondaryTransferStream);

    ForceCalculationKernel<<<secondaryBlocks * FACTOR5, THREADS5, 0>>>(nnodes, secondaryBodiesBegin, secondaryBodiesEnd, nbodies, secondary_errl, dthf, itolsq, epssq, secondary_sortl, secondary_childl, secondary_massl, secondary_posxl, secondary_posyl, secondary_poszl, secondary_velxl, secondary_velyl, secondary_velzl, secondary_accxl, secondary_accyl, secondary_acczl);


    cudaDeviceSynchronize();


    cudaSetDevice(mainDevice);
    cudaDeviceSynchronize();

    //Transfer only calculated acceleration back to main device
    if (cudaSuccess != cudaMemcpyPeer(main_temp_accxl, mainDevice, secondary_accxl, secondaryDevice, sizeof(double) * nbodies)) fprintf(stderr, "copying of accx from secondary device to main device failed\n");  //CudaTest("accx copy from device failed");
    if (cudaSuccess != cudaMemcpyPeer(main_temp_accyl, mainDevice, secondary_accyl, secondaryDevice, sizeof(double) * nbodies)) fprintf(stderr, "copying of accy from secondary device to main device failed\n");  //CudaTest("accy copy from device failed");
    if (cudaSuccess != cudaMemcpyPeer(main_temp_acczl, mainDevice, secondary_acczl, secondaryDevice, sizeof(double) * nbodies)) fprintf(stderr, "copying of accz from secondary device to main device failed\n");  //CudaTest("accz copy from device failed");

    //Copy updated acceleration information from temporary array to main array
    TransferKernel<<<mainBlocks * FACTOR7, THREADS7, 0>>>(secondaryBodiesBegin, secondaryBodiesEnd, main_sortl, main_accxl, main_accyl, main_acczl, main_temp_accxl, main_temp_accyl, main_temp_acczl);
    CudaTest("kernel 7 launch failed");

    IntegrationKernel<<<mainBlocks * FACTOR6, THREADS6, 0>>>(nbodies, dtime, dthf, main_posxl, main_posyl, main_poszl, main_velxl, main_velyl, main_velzl, main_accxl, main_accyl, main_acczl);
    CudaTest("kernel 6 launch failed");

    // transfer result back to CPU
    if (cudaSuccess != cudaMemcpy(&main_error     , main_errl      , sizeof(int)             , cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of err from main device failed\n");  //CudaTest("err copy from device failed");
    if (cudaSuccess != cudaMemcpy(&secondary_error, secondary_errl , sizeof(int)             , cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of err from secondary device failed\n");  //CudaTest("err copy from device failed");
    if (cudaSuccess != cudaMemcpy(posx            , main_posxl     , sizeof(double) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posx from main device failed\n");  //CudaTest("posx copy from device failed");
    if (cudaSuccess != cudaMemcpy(posy            , main_posyl     , sizeof(double) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posy from main device failed\n");  //CudaTest("posy copy from device failed");
    if (cudaSuccess != cudaMemcpy(posz            , main_poszl     , sizeof(double) * nbodies, cudaMemcpyDeviceToHost)) fprintf(stderr, "copying of posz from main device failed\n");  //CudaTest("posz copy from device failed");

    if(main_error || secondary_error){
      error = 1;
    }

    for(i = 0; i < nbodies; i++){
      fprintf(output, "%f " , posx[i]);
      fprintf(output, "%f " , posy[i]);
      fprintf(output, "%f\n", posz[i]);
    }
  }

  clock_t end = clock();

  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

  printf("Execution time: %lf\n", time_spent);

  if(error){
    printf("Execution failed\n");
  }else{
    printf("Execution successful\n");
  }

  fclose(output);

  cudaFreeHost(posx);
  cudaFreeHost(posy);
  cudaFreeHost(posz);


  cudaFree(main_errl);
  cudaFree(main_childl);
  cudaFree(main_massl);
  cudaFree(main_posxl);
  cudaFree(main_posyl);
  cudaFree(main_poszl);
  cudaFree(main_countl);
  cudaFree(main_startl);

  cudaFree(secondary_errl);
  cudaFree(secondary_childl);
  cudaFree(secondary_massl);
  cudaFree(secondary_posxl);
  cudaFree(secondary_posyl);
  cudaFree(secondary_poszl);
  cudaFree(secondary_countl);
  cudaFree(secondary_startl);

  cudaFree(maxxl);
  cudaFree(maxyl);
  cudaFree(maxzl);
  cudaFree(minxl);
  cudaFree(minyl);
  cudaFree(minzl);

  return 0;
}
