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
 * Filename: <parallel_cg.c>
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
#include <mpi.h>

#define EPSILON 1.0e-6                  /* exiting criteria   */                
#define ROWS 8192                       /* matrix rows        */                   
#define COLS 8192                       /* matrix columns     */                    
#define COL 1

void initialize(float *vector, char *filename, int colunm_num);
void matVec(float *local_matvec, float *local_matrixA,  
  float *local_vectorX,int local_row);
void residual(float *local_residual, float *local_vectorB,
  float *local_matvec, int local_row);
float vecVec(float *vect1,float *vect2, int local_row);
void scalarVec(float *scalar_vect, float *vect, 
  float scalar, int local_row);
void vecAdd(float *sumvect, float *vect1, 
  float *vect2, int local_row);
void vecSub(float *subvect, float *vect1, 
  float *vect2, int local_row);  
void conjugrad(float *local_matrixA,float *local_vectorB,
  float *local_vectorX, int local_row, int myrank, 
  int procsnum);
float allSum(int myrank,int procsnum, float temp_rsold, 
  float rsold);
float* memAllocate(float *vect, int m, int n);
void printer(float *vect, int rows1, int colunm_num);
int main(int argc, char* argv[])
{ 
  clock_t t;
  float *local_matrixA,			/* portion of matrix A   */
        *local_vectorX,			/* solution vector X     */
        *local_vectorB,			/* portion of vector B   */
        *matrixA,
        *vectorB;
  double start_time, 
         finish_time,
         cg_starttime,
         cg_finishtime,
         clock_time;
  int myrank, 
      procsnum,				/* total process number   */ 
      local_row;			/* rows of data           */
  MPI_Status status;
  local_matrixA = NULL;
  local_vectorX = NULL;
  local_vectorB = NULL;
  matrixA = NULL;
  vectorB = NULL;
  t = clock();
  if (MPI_Init (&argc, &argv) != MPI_SUCCESS) 
  {
    printf ("MPI_Init failed.\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &procsnum);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  local_row = ROWS / procsnum;
  if(myrank == 0)
  {
    if((ROWS % procsnum) != 0)
    {
      printf("%d is not divisible by %d\n",ROWS, procsnum);
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    if(ROWS != COLS)
    {
      printf("%d and %d must be same size\n",ROWS, COLS);
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }
  local_matrixA = memAllocate(local_matrixA, local_row, COLS);
  local_vectorX = memAllocate(local_vectorX, ROWS, COL);
  local_vectorB = memAllocate(local_vectorB, local_row, COL);
  if(myrank == 0)
  {
    printf("Computing cg of matrix size : %d\n", ROWS*COLS );
    matrixA = memAllocate(matrixA, ROWS, COLS);
    vectorB = memAllocate(vectorB, ROWS, COL);
    initialize(matrixA, argv[1], COLS);
    initialize(vectorB, argv[2], COL);
    initialize(local_vectorX, argv[3], COL);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  start_time = MPI_Wtime();
  MPI_Bcast(local_vectorX,ROWS*COL,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Scatter(matrixA,local_row*COLS,MPI_FLOAT,local_matrixA,
    local_row*COLS, MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Scatter(vectorB,local_row,MPI_FLOAT,local_vectorB,
    local_row, MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  finish_time = MPI_Wtime(); 
  conjugrad(local_matrixA, local_vectorB, local_vectorX, 
    local_row, myrank, procsnum);
  t = clock() - t;
  if(myrank == 0)
  {
    printf("collective data distribution time in seconds: %f\n", 
      (finish_time - start_time) );
    printf("clock execution time in seconds: %f\n", 
      ((double)t) / CLOCKS_PER_SEC );
    free(matrixA);
    free(vectorB);    
  }
  free(local_matrixA);
  free(local_vectorX);
  free(local_vectorB);
  MPI_Finalize();
  return 0;
}
/* memAllocate performs a memory allocation.    
 */
float* memAllocate(float *vect, int m, int n)
{
  if((vect = malloc ((m*n)*sizeof(float))) == NULL)
  {
    printf("can't allocate memory for vector\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  return vect;
}
void initialize(float *vector, char *filename, int col_num)
{
  FILE *reader;                         /* a file pointer         */
  int i,j;                              /* looping variables      */
  MPI_Status status;
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
 * Ax. The result is returned to local_matvec.
 */  
void matVec(float *local_matvec, float *local_matrixA, 
  float *local_vectorX, int local_row)
{
  int i,j;
  for(i = 0; i < local_row; ++i)
  {
    local_matvec[i] = 0.0;
    for(j = 0; j < COLS; ++j)
    {
      local_matvec[i] += local_matrixA[i*COLS+j] * local_vectorX[j];
    }
  } 
}
/* The residual function performs b-Ax. 
 * The result is returned to local_residual.
 */ 
void residual(float *local_residual, float *local_vectorB, 
  float *local_matvec, int local_row)
{
  int i;
  for(i=0; i < local_row; ++i)
  {
    local_residual[i] = local_vectorB[i] - local_matvec[i];
  }
}
/* scalarVec computes a scalar-vector multiplication. 
 * The result is stored to scalar_vect.
 */ 
void scalarVec(float *scalar_vect, float *vect, float scalar, 
  int local_row)
{
  int i;
  for(i=0; i < local_row; ++i)
  {
    scalar_vect[i] = vect[i] * scalar;
  }
}
/* vecVec returns the results of a vector-vector multiplication. 
 */ 
float vecVec(float *vect1,float *vect2, int local_row)
{
  int i;
  float sum;
  sum = 0.0;
  for (i=0; i < local_row; ++i)
  {
    sum += vect1[i] * vect2[i] ;
  }
  return sum;
}
/* vecAdd computes a vector-vector addition. 
 * The result is stored to sumvect.
 */ 
void vecAdd(float *sumvect, float *vect1, float *vect2, 
  int local_row)
{
  int i;
  for (i=0; i < local_row; ++i)
  {
    sumvect[i] = vect1[i] + vect2[i] ;
  } 
}
/* vecSub computes a vector-vector substraction. 
 * The result is stored to subvect.
 */ 
void vecSub(float *subvect, float *vect1, float *vect2, 
  int local_row)
{
  int i;
  for (i=0; i < local_row; ++i)
  {
    subvect[i] = vect1[i] - vect2[i] ;
  } 
}
/* main function for conjugate gradient method.    
 */
void conjugrad(float *local_matrixA,float *local_vectorB,
  float *local_vectorX, int   local_row, int   myrank, 
  int   procsnum)
{
  float *local_matvec,			/* pointer stores Ax       */			
        *local_vectorR,			/* portion of VectorR      */
        *local_vectorP,			/* portion of VectorP      */			
        *vectorP,			/* direction vector        */
        *vectorR,			/* residual  vector        */
        *temp_x,			/* temp vectorX            */
        *temp_r,			/* temp vectorR            */
        *temp_p,			/* temp vectorP            */
        rsold,				
        temp_rsold,
        temp_alpha,
        alpha,				/* step length             */
        temp_beta,
        beta;				/* new step length         */
  int   k;				/* loop variable           */
  double cg_starttime,
  	 cg_finishtime;
  MPI_Status status;
  local_matvec = memAllocate(local_matvec, local_row, COL);
  local_vectorR = memAllocate(local_vectorR, local_row, COL);
  local_vectorP = memAllocate(local_vectorP, local_row, COL);
  vectorP = memAllocate(vectorP, ROWS, COL);
  vectorR = memAllocate(vectorR, ROWS, COL);
  temp_x = memAllocate(temp_x, ROWS, COL);
  temp_r = memAllocate(temp_r, local_row, COL);	
  temp_p = memAllocate(temp_p, local_row, COL);
  MPI_Barrier(MPI_COMM_WORLD);
  cg_starttime = MPI_Wtime();
  
  /* start of conjugate gradient execution.    
   */
  matVec(local_matvec,local_matrixA,local_vectorX, local_row);
  residual(local_vectorR, local_vectorB, local_matvec, local_row);
  residual(local_vectorP, local_vectorB, local_matvec, local_row);
  temp_rsold = vecVec(local_vectorR, local_vectorR,  local_row);
  MPI_Allreduce(&temp_rsold,&rsold,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
  for(k=0; k < ROWS; ++k)
  {
    MPI_Allgather(local_vectorP, local_row, MPI_FLOAT, vectorP , local_row, 
      MPI_FLOAT,MPI_COMM_WORLD);
    matVec(local_matvec,local_matrixA,vectorP,local_row);
    temp_alpha = vecVec(local_vectorP, local_matvec,  local_row);
    MPI_Allreduce(&temp_alpha,&alpha,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    
    /* a step length to find an approximate solution    
     */
    alpha = rsold / alpha;
    scalarVec(temp_x, vectorP, alpha, ROWS);
    
    /* compute local_vectorX as approximate solution    
     */
    vecAdd(local_vectorX, local_vectorX, temp_x, ROWS);
    scalarVec(temp_r, local_matvec, alpha, local_row);
    
    /* compute local_vectorR as new residual    
     */
    vecSub(local_vectorR, local_vectorR, temp_r, local_row);
    temp_beta = vecVec(local_vectorR, local_vectorR,  local_row);
    
    /* an improved  step length.    
     */
    MPI_Allreduce(&temp_beta,&beta,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    if(sqrt(beta) < EPSILON)
    {
      break;
    }
    scalarVec(temp_p, local_vectorP, (beta / rsold), local_row);
    
    /* compute local_vectorP as new search direction    
     */
    vecAdd(local_vectorP, local_vectorR, temp_p, local_row);
    rsold = beta;
  }
  
  /* end of conjugate gradient execution.    
   */
  MPI_Barrier(MPI_COMM_WORLD);
  cg_finishtime = MPI_Wtime();
   
  if(myrank == 0)
  {
      //printer(local_vectorX, ROWS, 1);
      printf("cg method execution time in seconds: %f\n", 
        (cg_finishtime - cg_starttime) );  
  }      
  free(local_matvec);
  free(local_vectorR);
  free(local_vectorP);
  free(vectorP);
  free(vectorR);
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