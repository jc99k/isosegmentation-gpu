#ifndef THRUST_CUSTOMPREDICATES_H
#define THRUST_CUSTOMPREDICATES_H

#include "thrust_headers.h"

template<typename Tuple>
struct TuplePlus
{
  __host__ __device__
  Tuple operator()(Tuple x, Tuple y) const
  {
    using namespace thrust;
    return make_tuple(get<0>(x) + get<0>(y),
                      get<1>(x) + get<1>(y),
                      (get<2>(x)*get<1>(x) + get<2>(y)*get<1>(y))/(get<1>(x)+get<1>(y)));
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

//struct parallel_random_generator
//{
//	__host__ __device__
//		parallel_random_generator(unsigned int seed)
//	{
//		m_seed = seed;
//	}
//
//	__host__ __device__
//		float operator()(unsigned int n)
//	{
//		thrust::default_random_engine rng(m_seed);
//
//		// discard n numbers to avoid correlation
//		rng.discard(n);
//
//		// return a random number
//		return (float)rng() / thrust::default_random_engine::max;
//	}
//
//	unsigned int m_seed;
//};

#endif //THRUST_CUSTOMPREDICATES_H