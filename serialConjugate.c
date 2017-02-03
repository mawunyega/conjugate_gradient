#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define EPSILON 1.0e-7


// change all float to float

void initializeMatrixA(float *matrix, int rows, int cols, FILE *reader);

void initializeVectB(float *vect, int rows1, FILE *reader);

void initializeGuessVec(float *vect, int rows1, FILE *reader);

float residual(float *vectb, float *vectx, int rows1, float *temp);

float vecNorm(float *vectr, int rows1);

float vecVec(float *vect1,float *vect2, int rows1);

float matVec(float *matrix, int rows, int cols, float *vectr, float *tempvec);

float scalarVec(float *vect, float a, int rows1, float *temp);

float vecAdd(float *vect, float *vect1, int rows1, float *temp);

float vecSub(float *vect, float *vect1, int rows1, float *temp);

void printer(float *vect, int rows1);

int main(void)
{	


	FILE *reader;	
	
	time_t start,end;

	/* vectors to store matrices for computations*/
	float *matrixA;

	float *vectorb;
	
	
	/*initial guess , residual, and arbitary vectors of cg method*/
	float *vectorX;

	float *new_vectorX;

	float *vectorR;

	float *new_vectorR;

	float *vectorP;

	float *new_vectorP;

	float *tempvec;

	float *tempscalarvec;

	float *tempmatvec;	
	
	float alpha;

    	float beta;

    	float rho;

    	float store;

    	float temp_beta;

    


	/* rows and matrix columns variables*/
	int rows, cols, rows1,cols1,i;
	
	start = clock();

	reader = fopen("dimensions.txt", "r");

	if(reader != NULL)
	{
		fscanf(reader," %d %d %d %d\n", &rows, &cols, &rows1, &cols1); 

		fclose(reader);
	}

	else
	{
		printf("Could not open file\n");
	}

	printf("%d %d %d %d\n", rows, cols, rows1, cols1);


	
	if (cols != rows1)
	{
		printf("Matrix and vector dimensions do not match. Terminating program........\n");
		exit(0);

	}


	matrixA = malloc ((rows * cols)*sizeof(float) );

	vectorb = malloc ((rows1 * cols1)*sizeof(float) );

	vectorX = malloc (rows1 * sizeof(float) );

	new_vectorX = malloc (rows1 * sizeof(float) );

	vectorR = malloc (rows1 * sizeof(float) );

	new_vectorR = malloc (rows1 * sizeof(float) );

	vectorP = malloc (rows1 * sizeof(float) );

	new_vectorP = malloc (rows1 * sizeof(float) );
	
	tempvec = malloc (rows1 * sizeof(float) );

	tempmatvec = malloc ((rows1 * cols1)*sizeof(float) );
	
	tempscalarvec = malloc (rows1 * sizeof(float) );
	

	initializeMatrixA(matrixA, rows, cols, reader);

	
	initializeVectB(vectorb, rows1, reader);

	
	initializeGuessVec(vectorX, rows1,reader);


	matVec(matrixA, rows, cols, vectorX, tempmatvec);

	
	residual(vectorb,tempmatvec,rows1, vectorR);

	//printf("Ap prod is %f %f\n", vectorR[0], vectorR[1]);

	residual(vectorb,tempmatvec,rows1, vectorP);

	
	rho = vecVec(vectorR, vectorR, rows1);
	

	for(i = 0; i < rows1; i++)
	{
		

		matVec(matrixA,rows,cols,vectorP, tempvec);

		//printf("temVEC is %f %f %f %f\n", tempvec[0], tempvec[1], tempvec[2], tempvec[3]);

		store = vecVec(tempvec, vectorP, rows1);

		//printf("this is store %f\n",store);

		alpha = rho / store ;

		//printf("alph is %f\n", alpha);

		scalarVec(vectorP, alpha, rows1,tempscalarvec);

		//printf("alphaP is %f %f\n", tempscalarvec[0], tempscalarvec[1]);

		vecAdd(vectorX, tempscalarvec, rows1, vectorX);

		//printf("XX is %f %f\n", vectorX[0], vectorX[1]);

		scalarVec(tempvec, alpha, rows1,tempscalarvec);

		//printf("alphatempvec is %f %f\n", tempscalarvec[0], tempscalarvec[1]);

		//printf("vr111 is %f %f\n", vectorR[0], vectorR[1]);

		vecSub(vectorR, tempscalarvec, rows1, vectorR);

		//printf("RR is %f %f\n", vectorR[0], vectorR[1]);

		//tempstore = vecVec(new_vectorR,new_vectorR,rows1 );

		//printf("tempstore is %f\n", tempstore);

		//tempstore1 = vecVec(vectorR,vectorR,rows1 );

		beta = vecVec(vectorR,vectorR,rows1 );

		if(sqrt(beta) < EPSILON)
		{
			break;
		}

		temp_beta = beta / rho ;





		//printf("vr is %f %f\n", vectorR[0], vectorR[1]);

		//beta = tempstore / tempstore1 ;

		//printf("beta is %f\n", beta);

		//printf("tempstore1 is %f\n", tempstore1);

		scalarVec(vectorP, temp_beta, rows1, tempscalarvec);

		vecAdd(vectorR, tempscalarvec, rows1, vectorP);

		//printf("PP  is %f %f\n", vectorP[0], vectorP[1]);

		rho = beta;

		//printf("rho is %f\n", rho);

		

		//printf("rho is %f\n", rho);


		//printf("x before is %f %f\n", vectorX[0], vectorX[1]);

		//*vectorX = *new_vectorX;

		//printf("x after is %f %f\n", vectorX[0], vectorX[1]);

		//*vectorR = *new_vectorR;

		//*vectorP = *new_vectorP; 

		//printf("the vectr r is %f %f\n", vectorR[0], vectorR[1]);

		//printf("norm r is %f\n", vecNorm(vectorR,rows1)) ;    

	
	}
	
	end = clock();
	
	printf("clock execution time is %f\n",(end - start ) );


	printer(vectorX, rows1);
	


	free(matrixA);
	free(vectorb);
	free(vectorX);
	free(new_vectorX);
	free(vectorR);
	free(new_vectorR);
	free(vectorP);
	free(new_vectorP);
	free(tempvec);

	free(tempscalarvec);

	free(tempmatvec);



	return 0;

}

void initializeMatrixA(float *matrix, int rows, int cols, FILE *reader)
{
	int i,j;

	reader = fopen("matrixA_256X256.txt", "r");

	if(reader != NULL)
	{
		for(i=0; i<rows; i++)
		{
			for(j=0; j<cols; j++)
			{
				fscanf(reader,"%f%*c",&matrix[i*cols+j]);
			}
		}

		fclose(reader);

	}

	else
	{
		printf("Could not open file\n");
	}
	
	
}


void initializeVectB(float *vect, int rows1, FILE *reader)
{

	int i;

	reader = fopen("vectorb_256X1.txt", "r");

	if(reader != NULL)
	{
		for(i=0; i<rows1; i++)
		{
				fscanf(reader,"%f%*c",&vect[i]);
			
		}

		fclose(reader);

	}

	else
	{
		printf("Could not open filejjjjj\n");
	}
	

}

void initializeGuessVec(float *vect, int rows1, FILE *reader)
{

	int i;

	reader = fopen("X0_256X1.txt", "r");

	if(reader != NULL)
	{
		for(i=0; i<rows1; i++)
		{
				fscanf(reader,"%f%*c",&vect[i]);
			
		}

		fclose(reader);

	}

	else
	{
		printf("Could not open file\n");
	}
	

}

float matVec(float *matrix, int rows, int cols, float *vectr, float *tempvec)
{

	int i,j;
	

	for(i = 0; i < rows; i++)
	{
		tempvec[i] = 0.0;

		for(j = 0; j < cols; j++)
		{
			tempvec[i] = tempvec[i] + matrix[i*cols+j] * vectr[j];

		}
		
	}

	//printf("matvec prod is %f %f\n", tempvec[0], tempvec[1]);

	return *tempvec;	

}


float residual(float *vectb, float *vectx, int rows1, float *temp)
{
	int i;

	
	for(i=0; i<rows1; i++)
	{
		

		temp[i] = vectb[i] - vectx[i];
	}

	return *temp;

	
	
}

float vecNorm(float *vectr, int rows1)
{
	int i;

	float sum;

	sum = 0.0;

	for(i=0; i<rows1; i++)
	{
		sum += vectr[i] * vectr[i];
	}

	return sqrt(sum);
}


float vecVec(float *vect1,float *vect2, int rows1)
{
	int i;

	float sum;

	sum = 0.0;

	

	for (i=0; i < rows1; i++)
	{
		sum += vect1[i] * vect2[i] ;
	}

	return sum;


}

float vecAdd(float *vect, float *vect1, int rows1, float *temp)
{
	int i;

	for (i=0; i < rows1; i++)
	{
		temp[i] = vect[i] + vect1[i] ;
	}

	return *temp;

	

}

float vecSub(float *vect, float *vect1, int rows1, float *temp)
{
	int i;

	

	for (i=0; i < rows1; i++)
	{
		temp[i] = vect[i] - vect1[i] ;
	}

	return *temp;

	

}


float scalarVec(float *vect, float a, int rows1, float *temp)
{
	int i;

	for(i=0; i<rows1; i++)
	{
		temp[i] = vect[i] * a;
	}

	return *temp;


}

void printer(float *vect, int rows1)
{
	int i;

	for(i=0; i<rows1; i++)
	{
		printf("%f\n", vect[i]);

		
	}

}
