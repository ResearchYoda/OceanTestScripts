// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BuoyancySphereActor.generated.h"

class AWaterBody;
class UStaticMeshComponent;

/**
 * Queries Gerstner wave surface data (height, normal, velocity) from a WaterBodyLake
 * via QueryWaterInfoClosestToWorldLocation and applies Archimedes buoyancy to the sphere.
 */
UCLASS()
class FOSSENTEST_API ABuoyancySphereActor : public AActor
{
	GENERATED_BODY()

public:
	ABuoyancySphereActor();

protected:
	virtual void BeginPlay() override;
	virtual void Tick(float DeltaTime) override;

public:
	// ── Components ────────────────────────────────────────────────────────────

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	TObjectPtr<UStaticMeshComponent> SphereMesh;

	// ── Editor configuration ──────────────────────────────────────────────────

	/** WaterBodyLake actor to sample. Assign in the level Details panel. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Buoyancy")
	TObjectPtr<AWaterBody> WaterBody;

	/** Sphere radius in cm — match your mesh scale (engine Sphere.uasset = 50 cm at 1x). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Buoyancy", meta = (ClampMin = "1.0"))
	float SphereRadius = 50.0f;

	/** Water density kg/m³. Fresh water ~1000, sea water ~1025. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Buoyancy", meta = (ClampMin = "1.0"))
	float WaterDensity = 1025.0f;

	/** Scale the buoyancy force up or down. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Buoyancy", meta = (ClampMin = "0.0"))
	float BuoyancyMultiplier = 1.0f;

	/** Linear damping while submerged. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Buoyancy", meta = (ClampMin = "0.0"))
	float WaterLinearDamping = 3.0f;

	/** Angular damping while submerged. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Buoyancy", meta = (ClampMin = "0.0"))
	float WaterAngularDamping = 2.0f;

	/** Draw debug arrows and log each tick. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Buoyancy|Debug")
	bool bShowDebug = false;

	// ── Runtime read-outs ─────────────────────────────────────────────────────

	/** Absolute water surface height (Z, cm) at the sphere's XY position. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Buoyancy|Wave Info")
	float WaterSurfaceHeight = 0.f;

	/** Wave-tilted surface normal (includes Gerstner displacement). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Buoyancy|Wave Info")
	FVector WaterSurfaceNormal = FVector::UpVector;

	/** Water velocity (Gerstner orbital motion) at the sphere location. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Buoyancy|Wave Info")
	FVector WaterVelocity = FVector::ZeroVector;

	/** Wave amplitude (cm) at this location — from FWaveInfo.Height. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Buoyancy|Wave Info")
	float WaveAmplitude = 0.f;

	/**
	 * Dominant wave travel direction (XY, world space).
	 * Derived from the wave surface normal tilt: direction the surface faces away from vertical.
	 */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Buoyancy|Wave Info")
	FVector2D WaveDirection = FVector2D::ZeroVector;

	/** Fraction of sphere volume currently submerged [0, 1]. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Buoyancy|Wave Info")
	float SubmergedFraction = 0.f;

	/** Buoyancy force vector applied this frame (kg·cm/s²). */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Buoyancy|Wave Info")
	FVector AppliedBuoyancyForce = FVector::ZeroVector;

private:
	float DefaultLinearDamping  = 0.f;
	float DefaultAngularDamping = 0.f;
	bool  bWasSubmergedLastFrame = false;

	/** Query water surface info at WorldLocation. Returns false when outside the water body. */
	bool QueryWaterSurface(const FVector& WorldLocation);

	/** Spherical-cap submersion fraction using the stored WaterSurfaceHeight. */
	float ComputeSubmergedFraction(float SphereCentreZ) const;
};
