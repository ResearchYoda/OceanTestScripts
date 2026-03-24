// Copyright Epic Games, Inc. All Rights Reserved.

#include "BuoyancySphereActor.h"

#include "Components/StaticMeshComponent.h"
#include "DrawDebugHelpers.h"
#include "UObject/ConstructorHelpers.h"

// Water plugin — WaterBodyComponent.h pulls in WaterBodyTypes.h which has
// EWaterBodyQueryFlags, FWaterBodyQueryResult, and FWaveInfo.
#include "WaterBodyActor.h"
#include "WaterBodyComponent.h"

// ─────────────────────────────────────────────────────────────────────────────

ABuoyancySphereActor::ABuoyancySphereActor()
{
	PrimaryActorTick.bCanEverTick = true;

	SphereMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SphereMesh"));
	RootComponent = SphereMesh;

	SphereMesh->SetSimulatePhysics(true);
	SphereMesh->SetEnableGravity(true);
	SphereMesh->SetCollisionProfileName(TEXT("PhysicsActor"));

	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereMeshAsset(
		TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	if (SphereMeshAsset.Succeeded())
	{
		SphereMesh->SetStaticMesh(SphereMeshAsset.Object);
	}
}

// ─────────────────────────────────────────────────────────────────────────────

void ABuoyancySphereActor::BeginPlay()
{
	Super::BeginPlay();

	DefaultLinearDamping  = SphereMesh->GetLinearDamping();
	DefaultAngularDamping = SphereMesh->GetAngularDamping();
}

// ─────────────────────────────────────────────────────────────────────────────

void ABuoyancySphereActor::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!WaterBody)
	{
		return;
	}

	const FVector SphereLocation = GetActorLocation();

	// ── 1. Query Gerstner wave surface ────────────────────────────────────────
	if (!QueryWaterSurface(SphereLocation))
	{
		// Outside water body bounds — restore damping once.
		if (bWasSubmergedLastFrame)
		{
			SphereMesh->SetLinearDamping(DefaultLinearDamping);
			SphereMesh->SetAngularDamping(DefaultAngularDamping);
			bWasSubmergedLastFrame = false;
		}
		SubmergedFraction    = 0.f;
		AppliedBuoyancyForce = FVector::ZeroVector;
		return;
	}

	// ── 2. Submersion fraction (spherical cap formula) ────────────────────────
	SubmergedFraction = ComputeSubmergedFraction(SphereLocation.Z);
	const bool bSubmerged = SubmergedFraction > 0.f;

	// ── 3. Water damping ──────────────────────────────────────────────────────
	if (bSubmerged && !bWasSubmergedLastFrame)
	{
		SphereMesh->SetLinearDamping(WaterLinearDamping);
		SphereMesh->SetAngularDamping(WaterAngularDamping);
	}
	else if (!bSubmerged && bWasSubmergedLastFrame)
	{
		SphereMesh->SetLinearDamping(DefaultLinearDamping);
		SphereMesh->SetAngularDamping(DefaultAngularDamping);
	}
	bWasSubmergedLastFrame = bSubmerged;

	// ── 4. Buoyancy force (Archimedes) ────────────────────────────────────────
	if (bSubmerged)
	{
		// Radius and volume in SI (metres / m³).
		const float R_m           = SphereRadius * 0.01f;
		const float FullVol_m3    = (4.f / 3.f) * PI * R_m * R_m * R_m;
		const float SubVol_m3     = FullVol_m3 * SubmergedFraction;

		// UE gravity is in cm/s²; convert to m/s².
		const float Gravity_ms2   = FMath::Abs(GetWorld()->GetGravityZ()) * 0.01f;

		// Archimedes: F = ρ·V·g  [Newtons]
		// UE force units are kg·cm/s²; 1 N = 100 kg·cm/s²
		const float Force_N       = WaterDensity * SubVol_m3 * Gravity_ms2;
		const float Force_UE      = Force_N * 100.f * BuoyancyMultiplier;

		// Push along the wave-tilted surface normal, not just world Z.
		AppliedBuoyancyForce = WaterSurfaceNormal * Force_UE;
		SphereMesh->AddForce(AppliedBuoyancyForce, NAME_None, /*bAccelChange=*/false);

		// Water-current drag from Gerstner orbital velocity.
		if (!WaterVelocity.IsNearlyZero())
		{
			const FVector  RelVel      = WaterVelocity - SphereMesh->GetPhysicsLinearVelocity();
			const float    Speed_ms    = RelVel.Size() * 0.01f;
			const float    Cd          = 0.47f;                            // sphere
			const float    A_m2        = PI * (SphereRadius * 0.01f) * (SphereRadius * 0.01f);
			const float    Drag_N      = 0.5f * WaterDensity * Cd * A_m2 * Speed_ms * Speed_ms;
			SphereMesh->AddForce(RelVel.GetSafeNormal() * Drag_N * 100.f, NAME_None, false);
		}
	}
	else
	{
		AppliedBuoyancyForce = FVector::ZeroVector;
	}

	// ── 5. Debug ──────────────────────────────────────────────────────────────
	if (bShowDebug)
	{
		const FVector SurfacePoint(SphereLocation.X, SphereLocation.Y, WaterSurfaceHeight);

		DrawDebugSphere(GetWorld(), SurfacePoint, 8.f, 8, FColor::Cyan, false, -1.f, 0, 1.f);

		// Surface normal (blue)
		DrawDebugDirectionalArrow(GetWorld(), SurfacePoint,
			SurfacePoint + WaterSurfaceNormal * 80.f,
			15.f, FColor::Blue, false, -1.f, 0, 2.f);

		// Wave travel direction (green)
		if (!WaveDirection.IsNearlyZero())
		{
			DrawDebugDirectionalArrow(GetWorld(), SphereLocation,
				SphereLocation + FVector(WaveDirection.X, WaveDirection.Y, 0.f) * 120.f,
				15.f, FColor::Green, false, -1.f, 0, 2.f);
		}

		// Buoyancy force (yellow)
		if (bSubmerged)
		{
			DrawDebugDirectionalArrow(GetWorld(), SphereLocation,
				SphereLocation + AppliedBuoyancyForce.GetSafeNormal() * 100.f,
				15.f, FColor::Yellow, false, -1.f, 0, 2.f);
		}

		UE_LOG(LogTemp, Verbose,
			TEXT("[Buoyancy] SurfaceZ=%.1f | WaveAmp=%.1f cm | WaveDir=(%.2f,%.2f) | Submerged=%.2f | Force=%.0f N"),
			WaterSurfaceHeight, WaveAmplitude,
			WaveDirection.X, WaveDirection.Y,
			SubmergedFraction,
			AppliedBuoyancyForce.Size() / 100.f);
	}
}

// ─────────────────────────────────────────────────────────────────────────────

bool ABuoyancySphereActor::QueryWaterSurface(const FVector& WorldLocation)
{
	UWaterBodyComponent* Comp = WaterBody->GetWaterBodyComponent();
	if (!Comp)
	{
		return false;
	}

	// ComputeImmersionDepth is required for IsInWater().
	const EWaterBodyQueryFlags Flags =
		EWaterBodyQueryFlags::ComputeLocation         |
		EWaterBodyQueryFlags::ComputeNormal           |
		EWaterBodyQueryFlags::ComputeVelocity         |
		EWaterBodyQueryFlags::ComputeImmersionDepth   |
		EWaterBodyQueryFlags::IncludeWaves;

	const FWaterBodyQueryResult Result =
		Comp->QueryWaterInfoClosestToWorldLocation(WorldLocation, Flags);

	if (!Result.IsInWater())
	{
		return false;
	}

	// Absolute wave-displaced surface position and normal.
	WaterSurfaceHeight = Result.GetWaterSurfaceLocation().Z;
	WaterSurfaceNormal = Result.GetWaterSurfaceNormal();
	WaterVelocity      = Result.GetVelocity();

	// FWaveInfo carries the local wave amplitude and normal.
	const FWaveInfo& WaveInfo = Result.GetWaveInfo();
	WaveAmplitude = WaveInfo.Height;

	// Infer wave travel direction from the surface normal tilt.
	// A wave propagates in the direction the surface slopes downward → (-Nx, -Ny) normalised.
	const FVector2D NormalXY(WaterSurfaceNormal.X, WaterSurfaceNormal.Y);
	WaveDirection = NormalXY.IsNearlyZero() ? FVector2D::ZeroVector : -NormalXY.GetSafeNormal();

	return true;
}

// ─────────────────────────────────────────────────────────────────────────────

float ABuoyancySphereActor::ComputeSubmergedFraction(float SphereCentreZ) const
{
	// h = submersion depth measured from the bottom of the sphere.
	const float h = WaterSurfaceHeight - (SphereCentreZ - SphereRadius);

	if (h <= 0.f)       return 0.f;
	if (h >= 2.f * SphereRadius) return 1.f;

	// Spherical cap volume: V = π·h²·(R − h/3)
	const float R          = SphereRadius;
	const float CapVolume  = PI * h * h * (R - h / 3.f);
	const float FullVolume = (4.f / 3.f) * PI * R * R * R;

	return FMath::Clamp(CapVolume / FullVolume, 0.f, 1.f);
}
