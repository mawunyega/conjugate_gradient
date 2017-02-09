/* An MPI Conjugate Gradient Method solution for A*x = b. 
 * A is a dense NxN known matrix.
 * x is a dense NX1 unkown vector.
 * b is a dense NX1 known vector. 
 * 
 * Input parameters:matrixA_size.txt, vectorb_size.txt, X0_size.txt 
 *      
 * Execute file: first specify ROWS and COLS.
 * provide input file on command line in order above.
 *
 * Compile: mpicc filename.c -o filename -lm 
 * Run    : mpiexec -np 2 [--hosts host1,host2] filename para1.txt 
 *   para2.txt para3.txt
 *
 * Filename: <serialConjugate.c>
 * Author  : <Lloyd .M. Dzokoto>
 * Date    : <09.02.2017>
 * Version : <1>
 *
 */
 
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EPSILON 1.0e-6                  /* exiting criteria   */                
#define ROWS 8192                        /* matrix rows        */                   
#define COLS 8192                        /* matrix columns     */                    
#define COL 1

void initialize(float *vector, char *filename, int colunm_num);
void matVec(float *matvec, float *matrixA, float *vectorX);
void residual(float *residual, float *vectorB, float *matvec);
float vecVec(float *vect1,float *vect2);
void scalarVec(float *scalar_vect, float *vect, float scalar);
void vecAdd(float *sumvect, float *vect1, float *vect2);
void vecSub(float *subvect, float *vect1, float *vect2);  
void conjugrad(float *matrixA,float *vectorB,float *vectorX);
float* memAllocate(float *vect, int m, int n);
void printer(float *vect, int rows1, int colunm_num);
int main(int argc, char* argv[])
{ 
  float *matrixA,			/* portion of matrix A   */
        *vectorX,			/* solution vector X     */
        *vectorB;			/* portion of vector B   */
  if(argc != 4)
  {
    printf("serialCongugate.c requires four (4) files \n");
    exit (0);
  }    
  if(ROWS != COLS)
  {
    printf("%d and %d must be same size\n",ROWS, COLS);
    exit (0);
  }
  printf("Computing cg of matrix size : %d\n", ROWS*COLS );
  matrixA = NULL;
  vectorX = NULL;
  vectorB = NULL;
  matrixA = memAllocate(matrixA, ROWS, COLS);
  vectorX = memAllocate(vectorX, ROWS, COL);
  vectorB = memAllocate(vectorB, ROWS, COL);
  initialize(matrixA, argv[1], COLS);
  initialize(vectorB, argv[2], COL);
  initialize(vectorX, argv[3], COL);
  conjugrad(matrixA, vectorB, vectorX);
  free(matrixA);
  free(vectorX);
  free(vectorB);  
  return 0;
}
/* memAllocate performs a memory allocation.    
 */
float* memAllocate(float *vect, int m, int n)
{
  if((vect = malloc ((m*n)*sizeof(float))) == NULL)
  {
    printf("can't allocate memory for vector\n");
    exit (0);
  }
  return vect;
}
void initialize(float *vector, char *filename, int col_num)
{
  FILE *reader;                         /* a file pointer         */
  int i,j;                              /* looping variables      */
  reader = fopen(filename, "r");        /* open file for reading  */
  if(reader != NULL)                    
  {
    for(i = 0; i < ROWS; ++i)
    {
      for(j = 0; j < col_num; ++j)
      {
        fscanf(reader,"%f%*c", &vector[i*col_num+j]);
      }
    }
    fclose(reader);
  }
  else
  { 
    printf("Could not open file\n");
  } 
}
/* The matvec function requires a matrix and a vector to perform 
 * Ax. The result is returned to matvec.
 */  
void matVec(float *matvec, float *matrixA, float *vectorX)
{
  int i,j;
  for(i = 0; i < ROWS; ++i)
  {
    matvec[i] = 0.0;
    for(j = 0; j < COLS; ++j)
    {
      matvec[i] += matrixA[i*COLS+j] * vectorX[j];
    }
  } 
}
/* The residual function performs b-Ax. 
 * The result is returned to residual.
 */ 
void residual(float *residual, float *vectorB, float *matvec)
{
  int i;
  for(i=0; i < ROWS; ++i)
  {
    residual[i] = vectorB[i] - matvec[i];
  }
}
/* scalarVec computes a scalar-vector multiplication. 
 * The result is stored to scalar_vect.
 */ 
void scalarVec(float *scalar_vect, float *vect, float scalar)
{
  int i;
  for(i=0; i < ROWS; ++i)
  {
    scalar_vect[i] = vect[i] * scalar;
  }
}
/* vecVec returns the results of a vector-vector multiplication. 
 */ 
float vecVec(float *vect1,float *vect2)
{
  int i;
  float sum;
  sum = 0.0;
  for (i=0; i < ROWS; ++i)
  {
    sum += vect1[i] * vect2[i] ;
  }
  return sum;
}
/* vecAdd computes a vector-vector addition. 
 * The result is stored to sumvect.
 */ 
void vecAdd(float *sumvect, float *vect1, float *vect2)
{
  int i;
  for (i=0; i < ROWS; ++i)
  {
    sumvect[i] = vect1[i] + vect2[i] ;
  } 
}
/* vecSub computes a vector-vector substraction. 
 * The result is stored to subvect.
 */ 
void vecSub(float *subvect, float *vect1, float *vect2)
{
  int i;
  for (i=0; i < ROWS; ++i)
  {
    subvect[i] = vect1[i] - vect2[i] ;
  } 
}
/* main function for conjugate gradient method.    
 */
void conjugrad(float *matrixA,float *vectorB,float *vectorX)
{
  clock_t t;
  float *matvec,			/* pointer stores Ax       */			
        *vectorR,			/* residual  vector        */
        *vectorP,			/* direction vector        */			
        *temp_x,			/* temp vectorX            */
        *temp_r,			/* temp vectorR            */
        *temp_p,			/* temp vectorP            */
        rsold,				
        alpha,				/* step length             */
        beta;				/* new step length         */
  int   k;				/* loop variable           */
  matvec = NULL;
  vectorR = NULL;
  vectorP = NULL;
  temp_x = NULL;
  temp_r = NULL;
  temp_p = NULL;
  matvec = memAllocate(matvec, ROWS, COL);
  vectorR = memAllocate(vectorR, ROWS, COL);
  vectorP = memAllocate(vectorP, ROWS, COL);
  temp_x = memAllocate(temp_x, ROWS, COL);
  temp_r = memAllocate(temp_r, ROWS, COL);	
  temp_p = memAllocate(temp_p, ROWS, COL);
  
   /* start of conjugate gradient execution.    
   */
  t = clock();
  matVec(matvec,matrixA,vectorX);
  residual(vectorR, vectorB, matvec);
  residual(vectorP, vectorB, matvec);
  rsold = vecVec(vectorR, vectorR);
  for(k=0; k < ROWS; ++k)
  {
    matVec(matvec,matrixA,vectorP);
    
    /* a step length to find an approximate solution    
     */
    alpha = vecVec(vectorP, matvec);
    alpha = rsold / alpha;
    scalarVec(temp_x, vectorP, alpha);
    
    /* compute local_vectorX as approximate solution    
     */
    vecAdd(vectorX, vectorX, temp_x);
    scalarVec(temp_r, matvec, alpha);
    
    /* compute local_vectorR as new residual    
     */
    vecSub(vectorR, vectorR, temp_r);
    
    /* an improved  step length.    
     */
    beta = vecVec(vectorR, vectorR);
    if(sqrt(beta) < EPSILON)
    {
      break;
    }
    scalarVec(temp_p, vectorP, (beta / rsold));
    
    /* compute local_vectorP as new search direction    
     */
    vecAdd(vectorP, vectorR, temp_p);
    rsold = beta;
  }
  
  /* end of conjugate gradient execution.    
   */
  t = clock() - t;
  printf("average clock execution time in seconds: %f\n", 
    ((double)t) / CLOCKS_PER_SEC );   
  //printer(vectorX, ROWS, 1);       
  free(matvec);
  free(vectorR);
  free(vectorP);
  free(temp_x);
  free(temp_r);
  free(temp_p);
}
void printer(float *vect, int rows1, int colunm_num)
{
  int i, j;

  for(i=0; i < rows1; ++i)
  {
    for(j=0; j < colunm_num; ++j)
    {
      printf("%f  ", vect[i*colunm_num+j]);

      if(j == (colunm_num - 1)) 
        printf("\n");
    }
    
  }

}