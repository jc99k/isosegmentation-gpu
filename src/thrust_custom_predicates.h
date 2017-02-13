#ifndef THRUST_CUSTOMPREDICATES_H
#define THRUST_CUSTOMPREDICATES_H

#include "thrust_headers.h"
#include <cstdio>

template<typename Tuple>
struct TuplePlus
{
  __host__ __device__
  Tuple operator()(const Tuple &x, const Tuple &y) const
  {
	  Tuple temp;
	  using namespace thrust;
	  
	  get<0>(temp) = get<0>(x) + get<0>(y);
	  get<1>(temp) = get<1>(x) + get<1>(y);
	  float inv = 1.0 / get<1>(temp);
	  get<2>(temp) = get<2>(x)*(get<1>(x)*inv) + get<2>(y)*(get<1>(y)*inv);
	  return temp;
  }
};

template<typename Tuple>
struct TuplePlus2
{
	__host__ __device__
		Tuple operator()(const Tuple &x, const Tuple &y) const
	{
		Tuple temp;
		using namespace thrust;

		get<0>(temp) = get<0>(x) +get<0>(y);
		float inv = 1.0 / get<0>(temp);
		get<1>(temp) = get<1>(x)*(get<0>(x)*inv) + get<1>(y)*(get<0>(y)*inv);
		return temp;
	}
};

struct is_greater_than
{
  const int nR;
  is_greater_than(int _nR): nR(_nR) {}

  __host__ __device__
  bool operator()(const int &x) const
  {
    return x > nR;
  }
};

struct gamma_op
{
	const int v;
	const float vol;
	gamma_op(int _v, float _vol) : v(_v), vol(_vol) {}

	__host__ __device__
		float operator()(const int &nc, const float &ns) const
	{
		return (float(nc) / v) + (ns / vol);
	}
};

#endif //THRUST_CUSTOMPREDICATES_H