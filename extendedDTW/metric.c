#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "metric.h"


// custom distance function
// returns the square-root of the sum of the square of difference of the two values (elementwise)
double custom_dist(double *x, double *y, int numcols)
{
  int i;
  double dist;
  
  dist = 0;
  for( i=0; i<numcols; i++ )
  {  dist += pow(x[i]-y[i],2);
  }
  dist = sqrt( dist );

  return dist;
}
