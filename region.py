from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="specify_your_app_name_here", timeout = 3)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


region_arr = ['Arbutus-Ridge', 'Downtown', 'Dunbar-Southlands', 'Fairview', 'Grandview-Woodland', 'Hastings-Sunrise', 'Kensington-Cedar Cottage',
              'Kerrisdale', 'Killarney', 'Kitsilano', 'Marpole', 'Mount Pleasant', 'Oakridge', 'Renfrew-Collingwood', 'Riley Park', 'Shaughnessy',
              'South Cambie', 'Strathcona', 'Sunset', 'Victoria-Fraserview', 'West End', 'West Point Grey']

# Returns the region of provided address
# inputs
#   address: a string representing the address. input with format of:
#       "[Property_Postal_Code] [To_Civic_Number] [Street_Name]" (with the spaces)
# outputs
#   ret: a string representing the region
def getRegion(address):
    ret = "None"
    full_address = geolocator.geocode(address)
    address_array = str(full_address).split(',')
    for i in range(len(address_array)):
        if (address_array[i].strip() in region_arr):
            ret = address_array[i]
    return ret

print(getRegion('V5V 3S7 4642 WALDEN ST'))
print(getRegion('V6G 2S3 1477 PENDER ST W'))




