
#include <string.h>
#include "aes.h"
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda_runtime.h>
#include <cuda.h>


__constant__ uint8_t sbox_d[256]= {
		  //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
		  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
		  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
		  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
		  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
		  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
		  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
		  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
		  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
		  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
		  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
		  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
		  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
		  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
		  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
		  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
		  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };
__constant__ uint8_t rsbox_d[256] = {
		  0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
		  0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
		  0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
		  0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
		  0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
		  0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
		  0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
		  0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
		  0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
		  0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
		  0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
		  0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
		  0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
		  0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
		  0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
		  0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d };
__constant__ uint8_t Rcon_d[11] = {
		  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };
__constant__ int Nb_d = 4;
__constant__ int Nr_d = 14;
__constant__ int Nk_d = 8;
__constant__ uint32_t ek[60];

// The number of columns comprising a state in AES. This is a constant in AES. Value=4
#define Nb 4

#if defined(AES256) && (AES256 == 1)
    #define Nk 8
    #define Nr 14
#endif





typedef uint8_t state_t[4][4];

static const uint8_t sbox[256] = {
  //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };


const uint8_t Rcon[11] = {
  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

#define getSBoxValue(num) (sbox[(num)]);
#define device_getSBoxValue(num) (sbox_d[(num)]);



inline void cudaDevAssist(cudaError_t code, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"cudaDevAssistant: %s %d\n", cudaGetErrorString(code), line);
		if (abort) exit(code);
	}
}

static void KeyExpansion(uint8_t* RoundKey, const uint8_t* Key)
{
  unsigned i, j, k;
  uint8_t tempa[4]; // Used for the column/row operations

  // The first round key is the key itself.
  for (i = 0; i < Nk; ++i)
  {
    RoundKey[(i * 4) + 0] = Key[(i * 4) + 0];
    RoundKey[(i * 4) + 1] = Key[(i * 4) + 1];
    RoundKey[(i * 4) + 2] = Key[(i * 4) + 2];
    RoundKey[(i * 4) + 3] = Key[(i * 4) + 3];
  }

  // All other round keys are found from the previous round keys.
  for (i = Nk; i < Nb * (Nr + 1); ++i)
  {
    {
      k = (i - 1) * 4;
      tempa[0]=RoundKey[k + 0];
      tempa[1]=RoundKey[k + 1];
      tempa[2]=RoundKey[k + 2];
      tempa[3]=RoundKey[k + 3];

    }

    if (i % Nk == 0)
    {
      // This function shifts the 4 bytes in a word to the left once.
      // [a0,a1,a2,a3] becomes [a1,a2,a3,a0]

      // Function RotWord()
      {
        const uint8_t u8tmp = tempa[0];
        tempa[0] = tempa[1];
        tempa[1] = tempa[2];
        tempa[2] = tempa[3];
        tempa[3] = u8tmp;
      }

      // SubWord() is a function that takes a four-byte input word and
      // applies the S-box to each of the four bytes to produce an output word.

      // Function Subword()
      {
        tempa[0] = getSBoxValue(tempa[0]);
        tempa[1] = getSBoxValue(tempa[1]);
        tempa[2] = getSBoxValue(tempa[2]);
        tempa[3] = getSBoxValue(tempa[3]);
      }

      tempa[0] = tempa[0] ^ Rcon[i/Nk];
    }
#if defined(AES256) && (AES256 == 1)
    if (i % Nk == 4)
    {
      // Function Subword()
      {
        tempa[0] = getSBoxValue(tempa[0]);
        tempa[1] = getSBoxValue(tempa[1]);
        tempa[2] = getSBoxValue(tempa[2]);
        tempa[3] = getSBoxValue(tempa[3]);
      }
    }
#endif
    j = i * 4; k=(i - Nk) * 4;
    RoundKey[j + 0] = RoundKey[k + 0] ^ tempa[0];
    RoundKey[j + 1] = RoundKey[k + 1] ^ tempa[1];
    RoundKey[j + 2] = RoundKey[k + 2] ^ tempa[2];
    RoundKey[j + 3] = RoundKey[k + 3] ^ tempa[3];
  }
}

#if (defined(CTR) && (CTR == 1))
void AES_CTR_iv(struct AES_ctx* ctx, const uint8_t* key, const uint8_t* iv)
{
  KeyExpansion(ctx->RoundKey, key);
  memcpy (ctx->Iv, iv, AES_BLOCKLEN);
}
#endif

// This function adds the round key to state.
// The round key is added to the state by an XOR function.

__device__ void AddRoundKey(uint8_t round, state_t* myState, const uint8_t* RoundKey)
{
  uint8_t i,j;
  //state_t *devState = (state_t*)cipher;

  for (i = 0; i < 4; ++i)
  {
    for (j = 0; j < 4; ++j)
    {
      //(cipher)[i*4+j] ^= RoundKey[(round * Nb_d * 4) + (i * Nb_d) + j];
      (*myState)[i][j] ^= RoundKey[(round * Nb_d * 4) + (i * Nb_d) + j];
    	//(cipher)[i*4+j] = 'c';
    }
  }
}


__device__ void SubBytes(state_t* myState)
{
  uint8_t i, j;
  for (i = 0; i < 4; ++i)
  {
    for (j = 0; j < 4; ++j)
    {
      //(*devState)[j][i] = getSBoxValue((*devState)[j][i]);
      (*myState)[j][i] = sbox_d[(*myState)[j][i]];
    }
  }
}

// The ShiftRows() function shifts the rows in the state to the left.
// Each row is shifted with different offset.
// Offset = Row number. So the first row is not shifted.

__device__ void ShiftRows(state_t* myState)
{
  uint8_t temp;

  // Rotate first row 1 columns to left
  temp           = (*myState)[0][1];
  (*myState)[0][1] = (*myState)[1][1];
  (*myState)[1][1] = (*myState)[2][1];
  (*myState)[2][1] = (*myState)[3][1];
  (*myState)[3][1] = temp;

  // Rotate second row 2 columns to left
  temp           = (*myState)[0][2];
  (*myState)[0][2] = (*myState)[2][2];
  (*myState)[2][2] = temp;

  temp           = (*myState)[1][2];
  (*myState)[1][2] = (*myState)[3][2];
  (*myState)[3][2] = temp;

  // Rotate third row 3 columns to left
  temp           = (*myState)[0][3];
  (*myState)[0][3] = (*myState)[3][3];
  (*myState)[3][3] = (*myState)[2][3];
  (*myState)[2][3] = (*myState)[1][3];
  (*myState)[1][3] = temp;
}


__device__ uint8_t xtime(uint8_t x)
{
  return ((x<<1) ^ (((x>>7) & 1) * 0x1b));
}

// MixColumns function mixes the columns of the state matrix

__device__ void MixColumns(state_t* myState)
{
	uint8_t i;
	uint8_t Tmp, Tm, t;
	for (i = 0; i < 4; ++i)
	{
		t   = (*myState)[i][0];
		Tmp = (*myState)[i][0] ^ (*myState)[i][1] ^ (*myState)[i][2] ^ (*myState)[i][3] ;
		Tm  = (*myState)[i][0] ^ (*myState)[i][1] ;
		Tm = xtime(Tm); (*myState)[i][0] ^= Tm ^ Tmp ;
		Tm  = (*myState)[i][1] ^ (*myState)[i][2] ;
		Tm = xtime(Tm); (*myState)[i][1] ^= Tm ^ Tmp ;
		Tm  = (*myState)[i][2] ^ (*myState)[i][3] ;
		Tm = xtime(Tm); (*myState)[i][2] ^= Tm ^ Tmp ;
		Tm  = (*myState)[i][3] ^ t ;
		Tm = xtime(Tm); (*myState)[i][3] ^= Tm ^ Tmp ;
	}
}

// GPUCipher is the main function that encrypts the PlainText.
__global__ void GPUCipher(state_t* devState, const uint8_t* RoundKey, uint8_t* plain_text_d, state_t* myIv, int count) // HT
{
	int id = threadIdx.x;
	uint8_t round = 0;
	unsigned i;

	//uint8_t *myIv= (uint8_t *) (devState); // HT
	//state_t *myState = (state_t *) (myIv); // HT

//	for(int x = 0; x < 4; x++){
//		for(int y = 0; y < 4; y++){
//			(*myState)[x][y] = (*devState)[x][y]; // HT
//			//printf("My state: %d " ,myState[x][y]);
//		}
//	}
	AddRoundKey(0, devState, RoundKey); //devState -> myState
	for (round = 1; ; ++round)
	{
		SubBytes(devState); // getSBoxValue !!! // HT
		ShiftRows(devState); // HT
		if (round == Nr_d) {
		  break;
		}
		MixColumns(devState); // HT
		AddRoundKey(round, devState, RoundKey); // HT
	}

	 //Add round key to last round
	AddRoundKey(Nr_d, devState, RoundKey); // HT

	for(int a=0; a < id; ++a){
		for ( i = 0 ; i < 16; ++i)
		{
			/* inc will overflow */
			if (((uint8_t *)myIv)[i] == 255)
			{
				((uint8_t *)myIv)[i] = 0;
				continue;
			}
			//printf("%d", (int*)myIv[i]);
			((uint8_t *)myIv)[i] += 1;
			break;
		}
		plain_text_d[(id * 16) + i] = plain_text_d[(id * 16) + i] ^ ((uint8_t *)devState)[i];
//		//printf("Plaint text: %s ",(char*)plain_text_d);
	}
}
static void AddRoundKeyCPU(uint8_t round, state_t* state, const uint8_t* RoundKey)
{
  uint8_t i,j;
  for (i = 0; i < 4; ++i)
  {
    for (j = 0; j < 4; ++j)
    {
      (*state)[i][j] ^= RoundKey[(round * Nb * 4) + (i * Nb) + j];
    }
  }
}
static void SubBytesCPU(state_t* state)
{
  uint8_t i, j;
  for (i = 0; i < 4; ++i)
  {
    for (j = 0; j < 4; ++j)
    {
//      (*state)[j][i] = getSBoxValue((*state)[j][i]);
    	(*state)[j][i] = sbox[((*state)[j][i])];
    }
  }
}
static void ShiftRowsCPU(state_t* state)
{
  uint8_t temp;

  // Rotate first row 1 columns to left
  temp           = (*state)[0][1];
  (*state)[0][1] = (*state)[1][1];
  (*state)[1][1] = (*state)[2][1];
  (*state)[2][1] = (*state)[3][1];
  (*state)[3][1] = temp;

  // Rotate second row 2 columns to left
  temp           = (*state)[0][2];
  (*state)[0][2] = (*state)[2][2];
  (*state)[2][2] = temp;

  temp           = (*state)[1][2];
  (*state)[1][2] = (*state)[3][2];
  (*state)[3][2] = temp;

  // Rotate third row 3 columns to left
  temp           = (*state)[0][3];
  (*state)[0][3] = (*state)[3][3];
  (*state)[3][3] = (*state)[2][3];
  (*state)[2][3] = (*state)[1][3];
  (*state)[1][3] = temp;
}
static uint8_t xtimeCPU(uint8_t x)
{
  return ((x<<1) ^ (((x>>7) & 1) * 0x1b));
}
static void MixColumnsCPU(state_t* state)
{
  uint8_t i;
  uint8_t Tmp, Tm, t;
  for (i = 0; i < 4; ++i)
  {
    t   = (*state)[i][0];
    Tmp = (*state)[i][0] ^ (*state)[i][1] ^ (*state)[i][2] ^ (*state)[i][3] ;
    Tm  = (*state)[i][0] ^ (*state)[i][1] ; Tm = xtimeCPU(Tm);  (*state)[i][0] ^= Tm ^ Tmp ;
    Tm  = (*state)[i][1] ^ (*state)[i][2] ; Tm = xtimeCPU(Tm);  (*state)[i][1] ^= Tm ^ Tmp ;
    Tm  = (*state)[i][2] ^ (*state)[i][3] ; Tm = xtimeCPU(Tm);  (*state)[i][2] ^= Tm ^ Tmp ;
    Tm  = (*state)[i][3] ^ t ;              Tm = xtimeCPU(Tm);  (*state)[i][3] ^= Tm ^ Tmp ;
  }
}

static void CPUCipher(state_t* state, const uint8_t* RoundKey)
{
  uint8_t round = 0;

  // Add the First round key to the state before starting the rounds.
  AddRoundKeyCPU(0, state, RoundKey);

  // There will be Nr rounds.
  // The first Nr-1 rounds are identical.
  // These Nr rounds are executed in the loop below.
  // Last one without MixColumns()
  for (round = 1; ; ++round)
  {
    SubBytesCPU(state);
    ShiftRowsCPU(state);
    if (round == Nr) {
      break;
    }
    MixColumnsCPU(state);
    AddRoundKeyCPU(round, state, RoundKey);
  }
  // Add round key to last round
  AddRoundKeyCPU(Nr, state, RoundKey);
}



#if defined(CTR) && (CTR == 1)

/* Symmetrical operation: same function for encrypting as for decrypting. */
void AES_CTR_encryption(struct AES_ctx* ctx, uint8_t* buf, uint32_t length, int block_count , int count)
{

	uint8_t buffer[AES_BLOCKLEN];
	state_t *devState = NULL;
	uint8_t *roundKey_d = NULL;
	uint8_t *plain_text_d = NULL;
	state_t *myIv = NULL; // HT
	uint8_t *plain_text_h = NULL;
	uint8_t *buffer2;
	//uint8_t *buffer_d = NULL;
	unsigned i;
	int bi;
	if(count > 524288){ //GPU Encryption
		for (i = 0, bi = AES_BLOCKLEN; i < length; ++i, ++bi)
		{
			if (bi == AES_BLOCKLEN) /* we need to regen xor compliment in buffer */
			{

				//printf("\nENC: %s",(char*) buf);

				memcpy(buffer, ctx->Iv, AES_BLOCKLEN);
				//printf("\nbuffer: %s",(char*) buffer);
				cudaSetDevice(1);
	//			cudaDevAssist(cudaMemcpyToSymbol(Nk_d, &Nk, sizeof(int), 0, cudaMemcpyHostToDevice), 535, true);
	//			cudaDevAssist(cudaMemcpyToSymbol(Nr_d, &Nr, sizeof(int), 0, cudaMemcpyHostToDevice), 543, true);
	//			cudaDevAssist(cudaMemcpyToSymbol(Nb_d, &Nb, sizeof(int), 0, cudaMemcpyHostToDevice), 903, true);
	//			cudaDevAssist(cudaMemcpyToSymbol(sbox_d, &sbox, 256*sizeof(uint8_t), 0, cudaMemcpyHostToDevice), 920, true);
	//			cudaDevAssist(cudaMemcpyToSymbol(rsbox_d, &rsbox, 256*sizeof(uint8_t), 0, cudaMemcpyHostToDevice), 921, true);
	//			cudaDevAssist(cudaMemcpyToSymbol(Rcon_d, &Rcon, 11*sizeof(uint8_t), 0, cudaMemcpyHostToDevice), 922, true);
				//cudaThreadSynchronize();
				cudaDevAssist(cudaMalloc((void**)&devState, 16*sizeof(uint8_t)), 452, true);
				cudaDevAssist(cudaMalloc((void**)&roundKey_d, 240*sizeof(uint8_t)), 453, true);
				cudaDevAssist(cudaMalloc((void**)&plain_text_d, count*sizeof(uint8_t)),446, true);
				cudaDevAssist(cudaMalloc((void**)&myIv, 16*sizeof(uint8_t)),447, true);

				// Time starting
				cudaDevAssist(cudaMemcpy(devState, &buffer, 16*sizeof(uint8_t), cudaMemcpyHostToDevice), 455, true);
				cudaDevAssist(cudaMemcpy(myIv, &buffer, 16*sizeof(uint8_t), cudaMemcpyHostToDevice), 455, true);
				cudaDevAssist(cudaMemcpy(roundKey_d, ctx->RoundKey, 240*sizeof(uint8_t), cudaMemcpyHostToDevice), 456, true);
				cudaDevAssist(cudaMemcpy(plain_text_d, buf, count*sizeof(uint8_t), cudaMemcpyHostToDevice), 457, true);
				//printf("\nENC2: %s",(char*)roundKey_d);
				//cudaDevAssist(cudaMemcpy(buffer_d, buffer, textLength*sizeof(uint8_t), cudaMemcpyHostToDevice), 457, true);
				cudaDevAssist(cudaDeviceSynchronize(), 268, true);
				//cudaCipher<<<1,1>>>(devState,roundKey_d,buffer_d);
				GPUCipher<<<1,block_count>>>(devState,roundKey_d, plain_text_d,myIv,count);

				buffer2 = (uint8_t*)malloc(16*sizeof(uint8_t));
				plain_text_h = (uint8_t*)malloc(count*sizeof(uint8_t));
				cudaDevAssist(cudaMemcpy(buffer2, devState, 16*sizeof(uint8_t), cudaMemcpyDeviceToHost), 462, true);
				//cudaDevAssist(cudaMemcpy(plain_text_h, plain_text_d, count*sizeof(uint8_t), cudaMemcpyDeviceToHost), 463, true);
				//Time ending

				  /* Increment Iv and handle overflow */
//				for (bi = (AES_BLOCKLEN - 1); bi >= 0; --bi) //Önce hesapla sonra gönder.
//				{
//					/* inc will overflow */
//					if (ctx->Iv[bi] == 255)
//					{
//						ctx->Iv[bi] = 0;
//						continue;
//					}
//					ctx->Iv[bi] += 1;
//					break;
//				}
//				bi = 0;
			}

			//buf[i] = (buf[i] ^ buffer2[bi]);


		}
		cudaFree(devState);
		cudaFree(roundKey_d);
		cudaFree(plain_text_d);
		cudaFree(myIv);
		memcpy(buf, plain_text_h,count);
	}
	else{ //CPU Encryption
		for (i = 0, bi = AES_BLOCKLEN; i < length; ++i, ++bi)
		  {
		    if (bi == AES_BLOCKLEN) /* we need to regen xor compliment in buffer */
		    {

				  memcpy(buffer, ctx->Iv, AES_BLOCKLEN);
				  CPUCipher((state_t*)buffer,ctx->RoundKey);

				  /* Increment Iv and handle overflow */
				  for (bi = (AES_BLOCKLEN - 1); bi >= 0; --bi)
				  {
					/* inc will overflow */
						if (ctx->Iv[bi] == 255)
						{
						  ctx->Iv[bi] = 0;
						  continue;
						}
						ctx->Iv[bi] += 1; // 12A3  + 1 12A4 + 1
						break;
				  }
				  bi = 0;
		    }

		    buf[i] = (buf[i] ^ buffer[bi]); // buf = plain text, buffer = Iv
		  }
	}
}

#endif // #if defined(CTR) && (CTR == 1)
int main(int argc, const char * argv[])
{
	struct AES_ctx ctx;
	clock_t c_start, c_stop;

    uint8_t iv[16] = { 0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff };
    FILE *infile, *outfile, *keyfile;


	printf("Enter the path of input file");
	scanf("%s", argv[0]);
	printf("Enter the path of 32-bit key file");
	scanf("%s", argv[2]);






//    if(size % 16 != 0){
//    	int addition_part = 16 - (size % 16);
//		in = (uint8_t*)realloc(in,5*sizeof(int));
//		for (int i = size; i <= size + addition_part; ++i)
//			in[i+1] = 0;
//    }


    infile = fopen(argv[0], "r");
    fseek(infile, 0, SEEK_END);
	int size = ftell(infile);
	fseek(infile, 0, SEEK_SET);
	uint8_t in[size];
    int count = 0;
    for(int i = 0; i<size; ++i){
    	fread(&in, sizeof(char), size, infile);
    	count++;
    }
	fclose(infile);
	printf("\nData read from file: %s\n", in);

    keyfile = fopen(argv[2], "r");
	uint8_t key[32];
	fread(&key, sizeof(char), 32, keyfile);
	fclose(keyfile);
	printf("\nKeyfile: %s \n", key);


	int block_count = (count / 16) + (count % 16 == 0 ? 0 : 1); // 16 - count%16 kadar 0 eklenicek.


	printf("File size: %d bytes\n", size);

	outfile = fopen("/home/emre/cuda-workspace/AES256_CTR/src/output.txt", "w");
	// /home/emre/Desktop/Test_files/1kb.txt   /home/emre/cuda-workspace/AES256_CTR/src/key.txt
	int breaking_point = 524288; //This is the breaking point of our project. If file size is less than 512 kb, it runs on CPU, if it is larger than 512 kb, it runs on GPU.
	if (count >= breaking_point){

		printf("GPU initiliaze\n");
		printf("Data read from file: %s\n", in);
		AES_CTR_iv(&ctx, key, iv);
		c_start = clock();
		printf("Elements read: %d", count);
		printf("\nBlock Count: %d", block_count);
		AES_CTR_encryption(&ctx, in, strlen((char*)in),block_count,count);
		printf("\nENC: %s",(char*) in);
		fwrite(&in, sizeof(char), count, outfile);
		c_stop = clock();
		float diff = (((float)c_stop - (float)c_start) / CLOCKS_PER_SEC ) * 1000;
		printf("\nDone - Time taken on GPU: %f ms\n", diff);

	}
	else{
		printf("CPU initiliaze\n");


		//printf("keyfile: %s\n", key);
		printf("Elements read: %d\n", count);

		AES_CTR_iv(&ctx, key, iv);
		uint8_t Input[256];
		//uint8_t in2[size];
		printf("\nEnc:");
		c_start = clock();
		for(int i = 256, a=0; i<256+size;i+=256,a+=256){

			memcpy(Input, in+a,256);

			AES_CTR_encryption(&ctx, Input, strlen((char*)Input),block_count,count);
			printf("%s\n",(char*) Input); // don't use this string as an input
			fwrite(&Input, sizeof(char), 256, outfile);
		}
		c_stop = clock();
		float diff = (((float)c_stop - (float)c_start) / CLOCKS_PER_SEC ) * 1000;
		printf("\nDone - Time taken on CPU: %f ms\n", diff);
	}
	fclose(outfile);


    return 0;
}
