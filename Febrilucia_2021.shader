Shader "RedMage/Hallucidity/Febrilucia 2021" {
	Properties {
		[Enum(UnityEngine.Rendering.CullMode)] _CullMode("Cull Mode", Float) = 0
		[Enum(UnityEngine.Rendering.CompareFunction)] _ZTest ("ZTest", Int) = 4
		
        [Enum(A,0,B,1,C,2,D,3,Off,4)] _FixedPhase ("Fixed Phase", Int) = 4
        _DarkControl ("Darkness", Range(0, 1)) = 0
		_SlideyBoi ("Slide me 1", Range(-1, 1)) = 0
        _SlideyBoi2 ("Slide me 2", Range(-1, 1)) = 0
        _MainTex ("Main tesx", 2D) = "white" {}
        _BoopMap ("funny water texture thing go here", 2D) = "bump" {}
        _BoopQuantity ("how much water go boop", Float) = 0
        _MainTexBurnIn ("Main Tex Burn In", Range(0, 1)) = 0
		[Toggle(_)] _NotScreeny ("no screenspace bad bad bad", Int) = 0
	}
	SubShader {
		Tags { "RenderType"="Transparent" "Queue" = "Overlay+2" }
		
		Cull [_CullMode]
		ZTest [_ZTest]
		
		GrabPass { "_Garbasd" }

		Pass {
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"

			struct appdata {
				float4 vertex : POSITION;
			};

			struct v2f {
				float4 pos : SV_POSITION;
				float3 posWorld : TEXCOORD0;
				float4 projPos : TEXCOORD1;
			};

			sampler2D _MainTex;
			float4 _MainTex_ST;
			
			sampler2D _Garbasd;
			float4 _Garbasd_TexelSize;

            float _SlideyBoi, _SlideyBoi2;

            sampler2D _BoopMap;
            float4 _BoopMap_ST;

            uint _FixedPhase;

            float _BoopQuantity;

            float _DarkControl;
            float _MainTexBurnIn;

			uint _NotScreeny;

			float2 interpolateVec(float2 a, float2 b, float x) {
				float t0 = atan2(a.x, a.y);
				float t1 = atan2(b.x, b.y);

				float t;
				if (t0 < t1) {
					if (t1 - t0 > UNITY_PI) {
						t0 += UNITY_TWO_PI;
						t = t0 - x * (t0 - t1);	
					} else {
						t = t0 + x * (t1 - t0);
					}
				} else {
					if (t0 - t1 > UNITY_PI) {
						t1 += UNITY_TWO_PI;
						t = t0 + x * (t1 - t0);
					} else {
						t = t0 - x * (t0 - t1);
					}
				}
				return float2(cos(t), sin(t));
			}
			
			float3 hsv2rgb(float3 c) {
				return ((clamp(abs(frac(c.x+float3(0,.666,.333))*6-3)-1,0,1)-1)*c.y+1)*c.z;
			}
			
			float3 rgb2hsv(float3 c) {
				float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
				float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
				float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));

				float d = q.x - min(q.w, q.y);
				float e = 1.0e-10;
				return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
			}

            float2 mirrorSat(float2 uv) {
                if (uv.x < 0) uv.x = -uv.x;
                else if (uv.x > 1) uv.x = -1 + uv.x;
                
                if (uv.y < 0) uv.y = -uv.y;
                else if (uv.y > 1) uv.y = -1 + uv.y;
                return uv;
            }
			
			v2f vert (appdata v) {
				v2f o;
                float3 head = _WorldSpaceCameraPos.xyz;
				#if defined(USING_STEREO_MATRICES)
                head = lerp(unity_StereoWorldSpaceCameraPos[0], unity_StereoWorldSpaceCameraPos[1], .5);
                #endif

                if (unity_CameraProjection[2][0] != 0 || unity_CameraProjection[2][1] != 0) {
                    v.vertex = 1e25;
                } else {
                    v.vertex.xyz *= 8;
                }

                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
				o.pos = UnityObjectToClipPos(v.vertex);
				o.projPos = ComputeGrabScreenPos(o.pos);
				return o;
			}

            float mountains(float t) {
                float asdf = frac(t);
                if (asdf > .5) asdf = 1 - asdf;
                asdf *= 2;
                return asdf;
            }

            float3 colorBurn(float3 b, float3 s) {
                float3 res;
                UNITY_UNROLL for (int i = 0; i < 3; ++i) {
                    if (b[i] == 1) res[i] = 1;
                    else if (s[i] == 0) res[i] = 0;
                    else res[i] = 1 - min(1, (1 - b[i]) / s[i]);
                }
                return res;
            }

            float2x2 rotmat(float a) {
                float s = sin(a), c = cos(a);
                return float2x2(c, -s, s, c);
            }

            float3 baba(float3 co, float redthing, float time, float3 sstex, float snowFac) {
                co = rgb2hsv(co);

                float colorIntenser = 0;
                {
                    float a = frac(time * .6374 + .489539);
                    float c = co.x;
                    if (a > c) colorIntenser = min(a - c, 1 + c - a);
                    else colorIntenser = min(c - a, 1 + a - c);
                    colorIntenser = 1.2 * smoothstep(0, 0.5, colorIntenser);
                }

                co.x += redthing;
                co.x += .1 * sin(time / 2);
                co.x = frac(co.x);
                co.z = max(co.z * .5 * (1 - _DarkControl), pow(co.z, 1 + 1 * (.5 + .5*cos(time + .65))));
                co.z += colorIntenser * saturate(co.z);
                co.y = lerp(co.y, co.y * snowFac, .5 + .5 * sin(time * .923845 + 2.84593785));
				co.y += 0.2 * colorIntenser * saturate(co.y);
                co = hsv2rgb(co);

                co = lerp(co, colorBurn(co, sstex), _MainTexBurnIn * (.5 + .5 * sin(time * .12345666666666 + .34241234)));
                return co;
            }

			float2 computeSphereUV(float3 worldSpacePos) {
				float3 viewDir = normalize(worldSpacePos - _WorldSpaceCameraPos);
				float lat = acos(viewDir.y);
				float lon = atan2(viewDir.z, viewDir.x);
				lon = fmod(lon + UNITY_PI, UNITY_TWO_PI) - UNITY_PI;
				return 1 - float2(lon, lat) / UNITY_PI;
			}

            float3 calculatePass(float4 grabCol, float2 grabUV, float2 reallySSUV, float2 ssuv, float time, float snowFac, uint flip, uint cnt, out float3 prevColor, float finalStepInterpolant) {
                float redAcc = 0;
                float redPrev = 0;

                float3 col = 0;

                ssuv += 2 * (2 * grabCol.rg - 1);
                ssuv.y += frac(time * .214567);

				float2 centerProjUV = mul(UNITY_MATRIX_P, float4(.5, .5, 1, sin(1.238485903784 + .85673957 * time)));


                float3 boop_val = UnpackNormal(tex2Dlod(_BoopMap, float4(ssuv, 0, 0)));
                float2 distort = mul(rotmat(boop_val.z), boop_val.xy) * _BoopQuantity * saturate(sin(time * .1234589 + .234678));

                grabUV += distort;

				float gravitationTowardCenterAmt = .75 + .75 * sin(time * .48230573);
				//if (length(.5 - reallySSUV) < .1) gravitationTowardCenterAmt = 0;
				gravitationTowardCenterAmt *= smoothstep(.0, 1, length(.5 - reallySSUV));

                for (uint q = 0; q < cnt; ++q) {
                    redPrev = redAcc;
                    redAcc += grabCol.r;
					float3 hsv = rgb2hsv(grabCol.rgb);

					float focusFactor = 0;
					{
						float a = frac(time * .84938 + .193823889478);
						float c = hsv.x;
						if (a > c) focusFactor = min(a - c, 1 + c - a);
						else focusFactor = min(c - a, 1 + a - c);
						focusFactor = smoothstep(0, 0.5, focusFactor);
					}

					hsv.y = lerp(hsv.y, 1, .8 + .2 * sin(time * .394829458 + 101.3423895));
					hsv.z = lerp(hsv.z, 1, .8 + .2 * sin(time * .694829458 + 10.3423895));
                    grabCol.rgb = hsv2rgb(hsv);
                    float2 displace = normalize(grabCol.rg * 2 - 1);
					float2 towardCenterVec = normalize(centerProjUV - grabUV);
					displace = interpolateVec(displace, towardCenterVec, gravitationTowardCenterAmt);
                    displace *= .62 * .05 / cnt * (.75 + .25 * cos(time) + .5 * focusFactor);
                    displace += grabCol.b * .0005 * float2(sin(time), cos(time));
                    if (q == cnt - 1) displace *= finalStepInterpolant;

                    grabUV += displace;
                    grabUV = mirrorSat(grabUV);
                    grabUV = frac(grabUV);

                    float3 asdf = grabCol.rgb;
                    float abra = (flip & 1) != 0 ? -.5 : .5;
                    float kadabra = (flip & 2) != 0 ? -.5 : .5;

                    if (asdf.r * asdf.b < abra) {
                        grabUV.x = 1 - grabUV.x;
                    }
                    if (asdf.g < kadabra) {
                        grabUV.y = 1 - grabUV.y;
                    }
                    
                    grabCol = tex2D(_Garbasd, grabUV);
                    prevColor = col;
                    col += grabCol.rgb;
                }

                prevColor /= cnt - 1;
                redPrev /= cnt - 1;
                
                grabCol.rgb = col / cnt;
                redAcc /= cnt;

                float3 sstex = tex2Dlod(_MainTex, float4(ssuv + distort, 0, 0));
                //prevColor *= sstex;
                //grabCol.rgb *= sstex;
                prevColor = baba(prevColor, redPrev, time, sstex, snowFac);
                return baba(grabCol.rgb, redAcc, time, sstex, snowFac);
            }
            #define NOISE_SIMPLEX_1_DIV_289 0.00346020761245674740484429065744f

float mod289(float x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

float2 mod289(float2 x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

float3 mod289(float3 x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}

float4 mod289(float4 x) {
	return x - floor(x * NOISE_SIMPLEX_1_DIV_289) * 289.0;
}


// ( x*34.0 + 1.0 )*x = 
// x*x*34.0 + x
float permute(float x) {
	return mod289(
		x*x*34.0 + x
	);
}

float3 permute(float3 x) {
	return mod289(
		x*x*34.0 + x
	);
}

float4 permute(float4 x) {
	return mod289(
		x*x*34.0 + x
	);
}
float4 grad4(float j, float4 ip)
{
	const float4 ones = float4(1.0, 1.0, 1.0, -1.0);
	float4 p, s;
	p.xyz = floor( frac(j * ip.xyz) * 7.0) * ip.z - 1.0;
	p.w = 1.5 - dot( abs(p.xyz), ones.xyz );
	
	// GLSL: lessThan(x, y) = x < y
	// HLSL: 1 - step(y, x) = x < y
	p.xyz -= sign(p.xyz) * (p.w < 0);
	
	return p;
}


float snoise(float4 v)
{
	const float4 C = float4(
		0.138196601125011, // (5 - sqrt(5))/20 G4
		0.276393202250021, // 2 * G4
		0.414589803375032, // 3 * G4
	 -0.447213595499958  // -1 + 4 * G4
	);

// First corner
	float4 i = floor(
		v +
		dot(
			v,
			0.309016994374947451 // (sqrt(5) - 1) / 4
		)
	);
	float4 x0 = v - i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
	float4 i0;
	float3 isX = step( x0.yzw, x0.xxx );
	float3 isYZ = step( x0.zww, x0.yyz );
	i0.x = isX.x + isX.y + isX.z;
	i0.yzw = 1.0 - isX;
	i0.y += isYZ.x + isYZ.y;
	i0.zw += 1.0 - isYZ.xy;
	i0.z += isYZ.z;
	i0.w += 1.0 - isYZ.z;

	// i0 now contains the unique values 0,1,2,3 in each channel
	float4 i3 = saturate(i0);
	float4 i2 = saturate(i0-1.0);
	float4 i1 = saturate(i0-2.0);

	//	x0 = x0 - 0.0 + 0.0 * C.xxxx
	//	x1 = x0 - i1  + 1.0 * C.xxxx
	//	x2 = x0 - i2  + 2.0 * C.xxxx
	//	x3 = x0 - i3  + 3.0 * C.xxxx
	//	x4 = x0 - 1.0 + 4.0 * C.xxxx
	float4 x1 = x0 - i1 + C.xxxx;
	float4 x2 = x0 - i2 + C.yyyy;
	float4 x3 = x0 - i3 + C.zzzz;
	float4 x4 = x0 + C.wwww;

// Permutations
	i = mod289(i); 
	float j0 = permute(
		permute(
			permute(
				permute(i.w) + i.z
			) + i.y
		) + i.x
	);
	float4 j1 = permute(
		permute(
			permute(
				permute (
					i.w + float4(i1.w, i2.w, i3.w, 1.0 )
				) + i.z + float4(i1.z, i2.z, i3.z, 1.0 )
			) + i.y + float4(i1.y, i2.y, i3.y, 1.0 )
		) + i.x + float4(i1.x, i2.x, i3.x, 1.0 )
	);

// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
	const float4 ip = float4(
		0.003401360544217687075, // 1/294
		0.020408163265306122449, // 1/49
		0.142857142857142857143, // 1/7
		0.0
	);

	float4 p0 = grad4(j0, ip);
	float4 p1 = grad4(j1.x, ip);
	float4 p2 = grad4(j1.y, ip);
	float4 p3 = grad4(j1.z, ip);
	float4 p4 = grad4(j1.w, ip);

// Normalise gradients
	float4 norm = rsqrt(float4(
		dot(p0, p0),
		dot(p1, p1),
		dot(p2, p2),
		dot(p3, p3)
	));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;
	p4 *= rsqrt( dot(p4, p4) );

// Mix contributions from the five corners
	float3 m0 = max(
		0.6 - float3(
			dot(x0, x0),
			dot(x1, x1),
			dot(x2, x2)
		),
		0.0
	);
	float2 m1 = max(
		0.6 - float2(
			dot(x3, x3),
			dot(x4, x4)
		),
		0.0
	);
	m0 = m0 * m0;
	m1 = m1 * m1;
	
	return 49.0 * (
		dot(
			m0*m0,
			float3(
				dot(p0, x0),
				dot(p1, x1),
				dot(p2, x2)
			)
		) + dot(
			m1*m1,
			float2(
				dot(p3, x3),
				dot(p4, x4)
			)
		)
	);
}
			
float hash13(float3 p3)
{
	p3  = frac(p3 * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

			fixed4 frag (v2f i) : SV_Target {
				float3 viewSpace = mul(UNITY_MATRIX_V, i.posWorld.xyz - _WorldSpaceCameraPos);
                viewSpace.xy /= viewSpace.z;
				float2 grabUV = i.projPos.xy / i.projPos.w;
				float4 grabCol = tex2D(_Garbasd, grabUV);

                float width = _Garbasd_TexelSize.x;
                #if defined(USING_STEREO_MATRICES)
                width *= .5;
                #endif
                float height = _Garbasd_TexelSize.y;
                
                float2 ssuv = .5 * (1 - viewSpace.xy * float2((height*(width+1))/(width*(height+1)), 1));
				float2 reallySSUV = ssuv;

       			float snowFac = saturate(.8 + .2 * hash13(float3(floor(i.pos.xy), 1+frac(.745648+_Time.y * .463527))));

                float noiseFac = smoothstep(.4, 1.5 , length(ssuv - .5));
                float distortAmt = noiseFac * lerp(.2, 1, saturate(snoise(float4(ssuv * 2, 0, _Time.y))).x);


				if (_NotScreeny)
				{
					ssuv = computeSphereUV(i.posWorld.xyz);
				}

                float3 colll = 0;

                const uint cnt = 13;

                float timeParam =  _Time.y / 2;

                uint phaseCount = 4;

                float3 prevColor;

                float t = fmod(_Time.y * .3333333, phaseCount * 2);
                uint ft = (uint) (floor(t) + .5);

                if (_FixedPhase != 4) ft = ((uint) _FixedPhase) << 1;
                if ((ft & 1) == 0) {


                    // now we do the subphase transition
                    float tt = frac(t);
                    float subp = fmod(tt * cnt * 2, cnt * 2);
                    uint pp = (uint) (floor(subp) + .5);
                    float fac = frac(subp);
                    uint actualCnt = 0;
                    bool up = false;
                    if (pp < cnt) {
                        actualCnt = cnt + pp;
                    } else {
                        actualCnt = cnt + (2 * cnt - pp);
                        fac = 1 - fac;
                    }
                    uint phase = ft >> 1;
                    fac = smoothstep(0, 1, fac);
                    colll = calculatePass(grabCol, grabUV, reallySSUV, ssuv, timeParam, snowFac, phase, actualCnt, prevColor, fac);
                    
                    colll = lerp(prevColor, colll, fac);
                } else {
                    uint phase_a = ft >> 1;
                    uint phase_b = phase_a + 1;
                    float factor = frac(t);
                    float3 res_a = calculatePass(grabCol, grabUV, reallySSUV, ssuv, timeParam, snowFac, phase_a, cnt, prevColor, 1);
                    float3 res_b = calculatePass(grabCol, grabUV, reallySSUV, ssuv, timeParam, snowFac, phase_b, cnt, prevColor, 1);
                    colll = lerp(res_a, res_b, factor);
                }

                //colll = lerp(colll, float3(1, 1, 0), distortAmt);

                return float4(colll, 1);
			}
			ENDCG
		}
	}
}
