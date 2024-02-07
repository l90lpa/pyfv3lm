from pyfv3lm.settings import constants_set

# These constants have been copied from FMS's geos_constants.fh, gfdl_constants.fh, and gfs_constants.fh

#--- temporary definition for backwards compatibility
SMALL_FAC = 1.0

if constants_set == "geos":
    CONSTANTS_VERSION = "FMS Constants: GEOS"

    #--- Spherical coordinate conversion constants
    PI_8 = 3.14159265358979323846  #< Ratio of circle circumference to diameter [N/A]
    PI   = PI_8                    #< Ratio of circle circumference to diameter [N/A]
    RAD_TO_DEG  = 180.0/PI_8       #< Degrees per radian [deg/rad]
    DEG_TO_RAD  = PI_8/180.0       #< Radians per degree [rad/deg]
    RADIAN      = RAD_TO_DEG       #< Equal to RAD_TO_DEG for backward compatability. [rad/deg]
    
    #--- Earth physical constants
    RADIUS             = 6371.0E3           #< Radius of the Earth [m]
    OMEGA              = 2.0*PI_8/86164.0   #< Rotation rate of the Earth [1/s]
    GRAV               = 9.80665            #< Acceleration due to gravity [m/s^2]
    AGRAV              = 1.0 / GRAV         #< Reciprocal of acceleration due to gravity [s^2/m]
    SECONDS_PER_DAY    = 86400.0            #< Seconds in a day [s]
    SECONDS_PER_HOUR   =  3600.0            #< Seconds in an hour [s]
    SECONDS_PER_MINUTE =    60.0            #< Seconds in a minute [s]
    
    #--- Various gas constants
    RDGAS    = 8314.47 /28.965   #< Gas constant for dry air [J/kg/deg]
    RVGAS    = 8314.47 /18.015   #< Gas constant for water vapor [J/kg/deg]
    RDG      = -RDGAS * AGRAV
    HLV      = 2.4665E6            #< Latent heat of evaporation [J/kg]
    HLF      = 3.3370E5            #< Latent heat of fusion [J/kg]
    HLS      = HLV + HLF           #< Latent heat of sublimation [J/kg]
    KAPPA    = RDGAS/(3.5*RDGAS)   #< RDGAS / (3.5*RDGAS) [dimensionless]
    CP_AIR   = RDGAS/KAPPA         #< Specific heat capacity of dry air at constant pressure [J/kg/deg]
    CP_VAPOR = 4.0*RVGAS         #< Specific heat capacity of water vapor at constant pressure [J/kg/deg]
    CP_OCEAN = 3989.24495292815  #< Specific heat capacity taken from McDougall (2002) "Potential Enthalpy ..." [J/kg/deg]
    DENS_H2O = 1000.0            #< Density of liquid water [kg/m^3]
    RHOAIR   = 1.292269          #< Reference atmospheric density [kg/m^3]
    RHO0     = 1.035E3           #< Average density of sea water [kg/m^3]
    RHO0R    = 1.0/RHO0          #< Reciprocal of average density of sea water [m^3/kg]
    RHO_CP   = RHO0*CP_OCEAN     #< (kg/m^3)*(cal/kg/deg C)(joules/cal) = (joules/m^3/deg C) [J/m^3/deg]
    O2MIXRAT = 2.0953E-01        #< Mixing ratio of molecular oxygen in air [dimensionless]
    WTMAIR   = 2.896440E+01      #< Molecular weight of air [AMU]
    WTMH2O   = WTMAIR*(RDGAS/RVGAS)      #< Molecular weight of water [AMU]
    WTMOZONE =  47.99820         #< Molecular weight of ozone [AMU]
    WTMC     =  12.00000         #< Molecular weight of carbon [AMU]
    WTMCO2   =  44.00995         #< Molecular weight of carbon dioxide [AMU]
    WTMCH4   =  16.0425          #< Molecular weight of methane [AMU]
    WTMO2    =  31.9988          #< Molecular weight of molecular oxygen [AMU]
    WTMCFC11 = 137.3681          #< Molecular weight of CFC-11 (CCl3F) [AMU]
    WTMCFC12 = 120.9135          #< Molecular weight of CFC-21 (CCl2F2) [AMU]
    WTMN     =  14.0067          #< Molecular weight of Nitrogen [AMU]
    DIFFAC   = 1.660             #< Diffusivity factor [dimensionless]
    ES0      = 1.0               #< Humidity factor [dimensionless]. Controls the humidity content of
                                 #  the atmosphere through the Saturation Vapour Pressure expression 
                                 #  when using DO_SIMPLE
    
    #--- Pressure and Temperature constants
    PSTD     = 1.013250E+06      #< Mean sea level pressure [dynes/cm^2]
    PSTD_MKS = 101325.0          #< Mean sea level pressure [N/m^2]
    KELVIN   = 273.16            #< Degrees Kelvin at zero Celsius [K]
    TFREEZE  = 273.16            #< Freezing temperature of fresh water [K]
    C2DBARS  = 1.E-4             #< Converts rho*g*z (in mks) to dbars: 1dbar = 10^4 (kg/m^3)(m/s^2)m [dbars]
    
    #--- Named constants
    STEFAN   = 5.6734E-8         #< Stefan-Boltzmann constant [W/m^2/deg^4]
    AVOGNO   = 6.023000E+23      #< Avogadro's number [atoms/mole]
    VONKARM  = 0.40              #< Von Karman constant [dimensionless]
    
    #--- Miscellaneous constants
    ALOGMIN    = -50.0     #< Minimum value allowed as argument to log function [N/A]
    EPSLN      = 1.0E-40   #< A small number to prevent divide by zero exceptions [N/A]
    RADCON     = ((1.0E+02*GRAV)/(1.0E+04*CP_AIR))*SECONDS_PER_DAY  #< Factor to convert flux divergence
                                                                    #  to heating rate in degrees per day
                                                                    #  [deg sec/(cm day)]
    RADCON_MKS = (GRAV/CP_AIR)*SECONDS_PER_DAY #< Factor to convert flux divergence to heating rate
                                               #  in degrees per day [deg sec/(m day)]
    
elif constants_set == "gfdl":
    CONSTANTS_VERSION = "FMS Constants: GFDL"

    #--- Spherical coordinate conversion constants
    PI_8 = 3.14159265358979323846  #< Ratio of circle circumference to diameter [N/A]
    PI   = PI_8                    #< Ratio of circle circumference to diameter [N/A]
    RAD_TO_DEG  = 180./PI_8        #< Degrees per radian [deg/rad]
    DEG_TO_RAD  = PI_8/180.0       #< Radians per degree [rad/deg]
    RADIAN      = RAD_TO_DEG       #< Equal to RAD_TO_DEG for backward compatability. [rad/deg]

    #--- Earth physical constants
    RADIUS             = 6371.0E+3  #< Radius of the Earth [m]
    OMEGA              = 7.292E-5   #< Rotation rate of the Earth [1/s]
    GRAV               = 9.80       #< Acceleration due to gravity [m/s^2]
    AGRAV              = 1.0 / GRAV         #< Reciprocal of acceleration due to gravity [s^2/m]
    SECONDS_PER_DAY    = 86400.0    #< Seconds in a day [s]
    SECONDS_PER_HOUR   =  3600.0    #< Seconds in an hour [s]
    SECONDS_PER_MINUTE =    60.0    #< Seconds in a minute [s]

    #--- Various gas constants
    RDGAS    = 287.04            #< Gas constant for dry air [J/kg/deg]
    RVGAS    = 461.50            #< Gas constant for water vapor [J/kg/deg]
    RDG      = -RDGAS * AGRAV
    HLV      = 2.500E6           #< Latent heat of evaporation [J/kg]
    HLF      = 3.34E5            #< Latent heat of fusion [J/kg]
    HLS      = HLV + HLF         #< Latent heat of sublimation [J/kg]
    KAPPA    = 2.0/7.0           #< RDGAS / CP_AIR [dimensionless]
    CP_AIR   = RDGAS/KAPPA       #< Specific heat capacity of dry air at constant pressure [J/kg/deg]
    CP_VAPOR = 4.0*RVGAS         #< Specific heat capacity of water vapor at constant pressure [J/kg/deg]
    CP_OCEAN = 3989.24495292815  #< Specific heat capacity taken from McDougall (2002) "Potential Enthalpy ..." [J/kg/deg]
    DENS_H2O = 1000.0            #< Density of liquid water [kg/m^3]
    RHOAIR   = 1.292269          #< Reference atmospheric density [kg/m^3]
    RHO0     = 1.035E3           #< Average density of sea water [kg/m^3]
    RHO0R    = 1.0/RHO0          #< Reciprocal of average density of sea water [m^3/kg]
    RHO_CP   = RHO0*CP_OCEAN     #< (kg/m^3)*(cal/kg/deg C)(joules/cal) = (joules/m^3/deg C) [J/m^3/deg]
    O2MIXRAT = 2.0953E-01        #< Mixing ratio of molecular oxygen in air [dimensionless]
    WTMAIR   = 2.896440E+01      #< Molecular weight of air [AMU]
    WTMH2O   = WTMAIR*(RDGAS/RVGAS)      #< Molecular weight of water [AMU]
    WTMOZONE =  47.99820         #< Molecular weight of ozone [AMU]
    WTMC     =  12.00000         #< Molecular weight of carbon [AMU]
    WTMCO2   =  44.00995         #< Molecular weight of carbon dioxide [AMU]
    WTMCH4   =  16.0425          #< Molecular weight of methane [AMU]
    WTMO2    =  31.9988          #< Molecular weight of molecular oxygen [AMU]
    WTMCFC11 = 137.3681          #< Molecular weight of CFC-11 (CCl3F) [AMU]
    WTMCFC12 = 120.9135          #< Molecular weight of CFC-21 (CCl2F2) [AMU]
    WTMN     =  14.0067          #< Molecular weight of Nitrogen [AMU]
    DIFFAC   = 1.660             #< Diffusivity factor [dimensionless]
    ES0      = 1.0               #< Humidity factor [dimensionless] Controls the humidity content of
                                 #  the atmosphere through the Saturation Vapour Pressure expression
                                 #  when using DO_SIMPLE

    #--- Pressure and Temperature constants
    PSTD     = 1.013250E+06      #< Mean sea level pressure [dynes/cm^2]
    PSTD_MKS = 101325.0          #< Mean sea level pressure [N/m^2]
    KELVIN   = 273.15            #< Degrees Kelvin at zero Celsius [K]
    TFREEZE  = 273.16            #< Freezing temperature of fresh water [K]
    C2DBARS  = 1.E-4             #< Converts rho*g*z (in mks) to dbars: 1dbar = 10^4 (kg/m^3)(m/s^2)m [dbars]

    #--- Named constants
    STEFAN   = 5.6734E-8         #< Stefan-Boltzmann constant [W/m^2/deg^4]
    AVOGNO   = 6.023000E+23      #< Avogadro's number [atoms/mole]
    VONKARM  = 0.40              #< Von Karman constant [dimensionless]

    #--- Miscellaneous constants
    ALOGMIN    = -50.0        #< Minimum value allowed as argument to log function [N/A]
    EPSLN      = 1.0E-40      #< A small number to prevent divide by zero exceptions [N/A]
    RADCON     = ((1.0E+02*GRAV)/(1.0E+04*CP_AIR))*SECONDS_PER_DAY #< Factor to convert flux divergence
                                                                   #  to heating rate in degrees per day
                                                                   #  [deg sec/(cm day)]
    RADCON_MKS = (GRAV/CP_AIR)*SECONDS_PER_DAY #< Factor to convert flux divergence to heating rate 
                                               #  in degrees per day [deg sec/(m day)]

elif constants_set == "gfs":
    CONSTANTS_VERSION = "FMS Constants: GFS"

    #--- Spherical coordinate conversion constants
    PI_8 = 3.1415926535897931    #< Ratio of circle circumference to diameter [N/A]
    PI   = PI_8                  #< Ratio of circle circumference to diameter [N/A]
    RAD_TO_DEG  = 180.0/PI_8     #< Degrees per radian [deg/rad]
    DEG_TO_RAD  = PI_8/180.0     #< Radians per degree [rad/deg]
    RADIAN      = RAD_TO_DEG     #< Equal to RAD_TO_DEG for backward compatability. [rad/deg]

    #--- Earth physical constants
    RADIUS             = 6.3712E+6  #< Radius of the Earth [m]
    OMEGA              = 7.2921E-5  #< Rotation rate of the Earth [1/s]
    GRAV_8             = 9.80665    #< Acceleration due to gravity [m/s^2] (REAL(KIND=8))
    GRAV               = GRAV_8     #< Acceleration due to gravity [m/s^2]
    AGRAV              = 1.0 / GRAV         #< Reciprocal of acceleration due to gravity [s^2/m]
    SECONDS_PER_DAY    = 86400.0    #< Seconds in a day [s]
    SECONDS_PER_HOUR   =  3600.0    #< Seconds in an hour [s]
    SECONDS_PER_MINUTE =    60.0    #< Seconds in a minute [s]

    #--- Various gas constants
    RDGAS    = 287.05            #< Gas constant for dry air [J/kg/deg]
    RVGAS    = 461.50            #< Gas constant for water vapor [J/kg/deg]
    RDG      = -RDGAS * AGRAV
    HLV      = 2.500E6           #< Latent heat of evaporation [J/kg]
    HLF      = 3.3358e5          #< Latent heat of fusion [J/kg]
    HLS      = HLV + HLF         #< Latent heat of sublimation [J/kg]
    CP_AIR   = 1004.6            #< Specific heat capacity of dry air at constant pressure [J/kg/deg]
    CP_VAPOR = 4.0*RVGAS         #< Specific heat capacity of water vapor at constant pressure [J/kg/deg]
    CP_OCEAN = 3989.24495292815  #< Specific heat capacity taken from McDougall (2002) "Potential Enthalpy ..." [J/kg/deg]
    KAPPA    = RDGAS/CP_AIR      #< RDGAS / CP_AIR [dimensionless]
    DENS_H2O = 1000.0            #< Density of liquid water [kg/m^3]
    RHOAIR   = 1.292269          #< Reference atmospheric density [kg/m^3]
    RHO0     = 1.035E3           #< Average density of sea water [kg/m^3]
    RHO0R    = 1.0/RHO0          #< Reciprocal of average density of sea water [m^3/kg]
    RHO_CP   = RHO0*CP_OCEAN     #< (kg/m^3)*(cal/kg/deg C)(joules/cal) = (joules/m^3/deg C) [J/m^3/deg]
    O2MIXRAT = 2.0953E-01        #< Mixing ratio of molecular oxygen in air [dimensionless]
    WTMAIR   = 2.896440E+01      #< Molecular weight of air [AMU]
    WTMH2O   = WTMAIR*(RDGAS/RVGAS)      #< Molecular weight of water [AMU]
    WTMOZONE =  47.99820         #< Molecular weight of ozone [AMU]
    WTMC     =  12.00000         #< Molecular weight of carbon [AMU]
    WTMCO2   =  44.00995         #< Molecular weight of carbon dioxide [AMU]
    WTMCH4   =  16.0425          #< Molecular weight of methane [AMU]
    WTMO2    =  31.9988          #< Molecular weight of molecular oxygen [AMU]
    WTMCFC11 = 137.3681          #< Molecular weight of CFC-11 (CCl3F) [AMU]
    WTMCFC12 = 120.9135          #< Molecular weight of CFC-21 (CCl2F2) [AMU]
    WTMN     =  14.0067          #< Molecular weight of Nitrogen [AMU]
    DIFFAC   = 1.660             #< Diffusivity factor [dimensionless]
    ES0      = 1.0               #< Humidity factor [dimensionless] Controls the humidity content of
                                 #  the atmosphere through the Saturation Vapour Pressure expression
                                 #  when using DO_SIMPLE
    CON_CLIQ = 4.1855E+3         #< Specific heat H2O liq [J/kg/K]
    CON_CSOL = 2.1060E+3         #< Specific heat H2O ice [J/kg/K]

    #--- Pressure and Temperature constants
    PSTD     = 1.013250E+06      #< Mean sea level pressure [dynes/cm^2]
    PSTD_MKS = 101325.0          #< Mean sea level pressure [N/m^2]
    KELVIN   = 273.15            #< Degrees Kelvin at zero Celsius [K]
    TFREEZE  = 273.15            #< Freezing temperature of fresh water [K]
    C2DBARS  = 1.E-4             #< Converts rho*g*z (in mks) to dbars: 1dbar = 10^4 (kg/m^3)(m/s^2)m [dbars]

    #--- Named constants
    STEFAN   = 5.6734E-8         #< Stefan-Boltzmann constant [W/m^2/deg^4]
    AVOGNO   = 6.023000E+23      #< Avogadro's number [atoms/mole]
    VONKARM  = 0.40              #< Von Karman constant [dimensionless]

    #--- Miscellaneous constants
    ALOGMIN    = -50.0        #< Minimum value allowed as argument to log function [N/A]
    EPSLN      = 1.0E-40      #< A small number to prevent divide by zero exceptions [N/A]
    RADCON     = ((1.0E+02*GRAV)/(1.0E+04*CP_AIR))*SECONDS_PER_DAY #< Factor to convert flux divergence
                                                                   #  to heating rate in degrees per day
                                                                   #  [deg sec/(cm day)]
    RADCON_MKS = (GRAV/CP_AIR)*SECONDS_PER_DAY #< Factor to convert flux divergence to heating rate 
                                               #  in degrees per day [deg sec/(m day)]
else:
    raise ValueError(f"Unsupported constants set {constants_set}.")