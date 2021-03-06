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
 * Filename: <point-to-point_cg.c>
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
void scatterRow(float *local_matrix, char *filename, 
  int local_row, int myrank, int procsnum, int col);
void BcastVector(float *local_vectorX,int myrank, 
  int procsnum, int cnt); 
float allSum(int myrank,int procsnum, float temp_rsold, 
  float rsold);
void allGather(float *vect, float *vect1,int local_row,
  int myrank,int procsnum);
float* memAllocate(float *vect, int m, int n);
void printer(float *vect, int rows1, int colunm_num);

int main(int argc, char* argv[])
{ 
  clock_t t;
  float *local_matrixA,			/* portion of matrix A   */
        *local_vectorX,			/* solution vector X     */
        *local_vectorB;			/* portion of vector B   */
        
  double start_time, 
         finish_time,
         clock_time;
         
  int myrank, 
      procsnum,				/* total process number   */ 
      local_row;			/* rows of data           */
      
  MPI_Status status;
  t = clock();
  /* MPI initialization begins 
   */
  if (MPI_Init (&argc, &argv) != MPI_SUCCESS) 
  {
    printf ("MPI_Init failed.\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  
  local_matrixA = NULL;
  local_vectorX = NULL;
  local_vectorB = NULL;
  
  /* Get number of processes and own rank
   */
  MPI_Comm_size(MPI_COMM_WORLD, &procsnum);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  
  /* processes calculate required number of rows.
   */
  local_row = ROWS / procsnum;
  if(myrank == 0)
  {
    if((ROWS % procsnum) != 0)
    {
      printf("%d must be divisible by %d\n",ROWS, procsnum);
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    if(ROWS != COLS)
    {
      printf("%d and %d must be same size\n",ROWS, COLS);
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }
  local_row = ROWS / procsnum;
  local_matrixA = memAllocate(local_matrixA, local_row, COLS);
  local_vectorX = memAllocate(local_vectorX, ROWS, COL);
  local_vectorB = memAllocate(local_vectorB, local_row, COL);
  if(myrank == 0)
  {
    printf("Computing cg of matrix size : %d\n", ROWS*COLS );
    initialize(local_vectorX, argv[3], COL);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  start_time = MPI_Wtime();
  BcastVector(local_vectorX,myrank, procsnum, ROWS*COL);
  scatterRow(local_matrixA, argv[1], local_row, 
    myrank, procsnum, COLS);  
  scatterRow(local_vectorB, argv[2], local_row, 
    myrank, procsnum, COL);
  MPI_Barrier(MPI_COMM_WORLD);
  finish_time = MPI_Wtime(); 
  conjugrad(local_matrixA, local_vectorB, local_vectorX, 
    local_row, myrank, procsnum);
  t = clock() - t;
  if(myrank == 0)
  {
    printf("p2p data distribution time in seconds: %f\n", 
      (finish_time - start_time) );
    printf("clock execution time in seconds: %f\n", 
      ((double)t) / CLOCKS_PER_SEC );    
  }
  free(local_matrixA);
  free(local_vectorX);
  free(local_vectorB);
  
  /* MPI cleaning up. 
   */
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
/* scatterRow emulates MPI_Scatter using MPI_Send and MPI_Recv.   
 * Process 0 sends data of local_row*col size to all processes. 
 */
void scatterRow(float *local_matrix, char *filename, 
  int local_row, int myrank, int procsnum, int col)
{
  FILE *reader;
  int dest,i,j;
  float *temp;
  MPI_Status status;
  temp = memAllocate(temp,local_row,col);
  if(myrank == 0)
  {
    reader = fopen(filename, "r");
    if (reader != NULL)
    {
      /* process rank 0 reads the first local_row by col size
       * of data and as its row strip of the matrix. 
       */
      for(i = 0; i < local_row; ++i)
      {
        for(j=0; j < col; ++j)
        {
          fscanf(reader, "%f%*c", &local_matrix[i*col+j]);
        }
      }
      /* process rank 0 reads a local_row by col size of data and 
       * sends to the rest of process. This gives each process its 
       * row strip of the matrix.
       */  
      for (dest = 1; dest < procsnum; ++dest) 
      {
        for (i = 0; i < local_row; ++i)
        {
          for(j=0; j < col; ++j)
          {
            fscanf(reader, "%f%*c", &temp[i*col+j]);
          }
        }
        MPI_Send(temp,local_row*col, MPI_FLOAT, dest, 0, 
          MPI_COMM_WORLD);  
      }
      fclose(reader);
    }
    else
    {
      printf("Could not open %s file. \n", filename);
    }    
  }
  /* every other process receives their row strip of the matrix.*/
  else 
   {
      MPI_Recv(local_matrix, local_row*COLS, MPI_FLOAT,0 , 
        0, MPI_COMM_WORLD, &status);
   }
}
/* BcastVector function emulates MPI_Bcast using MPI_Send and MPI_Recv. 
 * It distributes local_vectorX to  all processes from process 0. 
 */
void BcastVector(float *local_vectorX, int myrank, int procsnum, 
  int cnt)
{
  int dest;
  MPI_Status status;
  if(myrank == 0)
  {
    for(dest = 1; dest < procsnum; ++dest)
    {
      MPI_Send(local_vectorX, cnt,MPI_FLOAT,dest,1, MPI_COMM_WORLD);
    }
  }
  else
  {
    MPI_Recv(local_vectorX, cnt,MPI_FLOAT,0,1, MPI_COMM_WORLD, 
      &status);
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
/* allSum emulates MPI_Reduce using MPI_Send and MPI_Recv.    
 * temp_rsold is sent from each process and a summation 
 * performed by process 0. rsold is returned
 * as the sum total of temp_rsold. 
 */
float allSum(int myrank,int procsnum, float temp_rsold, 
  float rsold)
{
  int dest;
  MPI_Status status;
  if(myrank == 0)
  {
    rsold = temp_rsold;
    for(dest = 1; dest < procsnum; ++dest)
    {
      MPI_Recv(&temp_rsold,1,MPI_FLOAT,dest,2,MPI_COMM_WORLD,
        &status);
      rsold += temp_rsold;
    }
  }
  else
  {
    MPI_Send(&temp_rsold,1,MPI_FLOAT,0,2,MPI_COMM_WORLD);
  }
  return rsold;  
}
/* allGather emulates MPI_Gather using MPI_Send and MPI_Recv.    
 * vect1 is sent from each process and its stored in process 
 * rank order on vect on process 0.
 */
void allGather(float *vect, float *vect1,int local_row,int myrank,
  int procsnum)
{
  int dest,i,index, rank;
  float *temp_localP;
  temp_localP = memAllocate(temp_localP, local_row, COL);
  index = local_row;
  MPI_Status status;
  if(myrank == 0)
  {
    for(i=0; i < local_row; ++i)
    {
      vect[i] = vect1[i];
    }
    for(dest = 1; dest < procsnum; ++dest)
    {
      MPI_Recv(temp_localP, local_row, MPI_FLOAT, dest, 3, 
        MPI_COMM_WORLD, &status);
      for(i = 0; i < local_row; ++i)
      {
        vect[index] = temp_localP[i];
        index++;
      }
    }
  }
  else
  {
    MPI_Send(vect1,local_row,MPI_FLOAT,0,3,MPI_COMM_WORLD);
  }
  free(temp_localP);   
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
        rsold_sum,
        temp_rsold,
        temp_alpha,
        alpha,				/* step length             */
        alpha_sum,
        temp_beta,
        beta,				/* new step length         */
        beta_sum;
  int   k,				/* loop variable           */
        q;
  double cg_starttime,
  	 cg_finishtime;		        /* number of bytes         */
  rsold_sum = 0.0;
  alpha_sum = 0.0;
  beta_sum = 0.0;
  q = 1;
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
  rsold = allSum(myrank,procsnum, temp_rsold,rsold_sum);
  BcastVector(&rsold, myrank, procsnum, q);
  for(k=0; k < ROWS; ++k)
  {
    allGather(vectorP, local_vectorP,local_row,myrank,procsnum);
    BcastVector(vectorP, myrank, procsnum, ROWS);
    matVec(local_matvec,local_matrixA,vectorP,local_row);
    temp_alpha = vecVec(local_vectorP, local_matvec,  local_row);
    
    /* a step length to find an approximate solution    
     */
    alpha = allSum(myrank,procsnum, temp_alpha,alpha_sum);
    BcastVector(&alpha, myrank, procsnum, q);
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
    beta = allSum(myrank,procsnum, temp_beta,beta_sum);
    BcastVector(&beta, myrank, procsnum, q);
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
    for(j= 0; j < colunm_num; ++j)
    {
      printf("%f ", vect[i*colunm_num+j]); 
    }
    printf("\n");
  }

}