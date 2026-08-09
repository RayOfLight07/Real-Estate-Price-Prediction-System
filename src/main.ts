import { gsap } from 'gsap';
import L from 'leaflet';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

// Interface Definitions
interface PredictionResponse {
  success: boolean;
  price_lakhs: number;
  price_crores: number;
  formatted_price: string;
  price_range?: string;
  rate_per_sqft: number;
  bhk: number;
  property_type: string;
  city: string;
  state: string;
  error?: string;
}


interface MetadataResponse {
  states: string[];
  cities_by_state: Record<string, string[]>;
  localities_by_city: Record<string, string[]>;
  property_types: string[];
  furnishing_options: string[];
  facing_options: string[];
  owner_options: string[];
  availability_options: string[];
  transport_options: string[];
}

interface AnalyticsResponse {
  city_stats: Record<string, {
    state: string;
    lat: number;
    lon: number;
    avg_price_lakhs: number;
    avg_sqft: number;
    avg_rate_per_sqft: number;
    total_listings: number;
  }>;
  state_stats: Record<string, {
    avg_price_lakhs: number;
    avg_rate_per_sqft: number;
    total_listings: number;
  }>;
  bhk_stats: Record<string, number>;
  prop_type_stats: Record<string, number>;
}

// 42 City Coordinate Database fallback
const CITY_COORDINATES: Record<string, { state: string; lat: number; lon: number }> = {
  "Vijayawada": { state: "Andhra Pradesh", lat: 16.5062, lon: 80.6480 },
  "Vishakhapatnam": { state: "Andhra Pradesh", lat: 17.6868, lon: 83.2185 },
  "Guwahati": { state: "Assam", lat: 26.1445, lon: 91.7362 },
  "Silchar": { state: "Assam", lat: 24.8333, lon: 92.7789 },
  "Gaya": { state: "Bihar", lat: 24.7914, lon: 85.0002 },
  "Patna": { state: "Bihar", lat: 25.5941, lon: 85.1376 },
  "Bilaspur": { state: "Chhattisgarh", lat: 22.0797, lon: 82.1391 },
  "Raipur": { state: "Chhattisgarh", lat: 21.2514, lon: 81.6296 },
  "Dwarka": { state: "Delhi", lat: 28.5921, lon: 77.0460 },
  "New Delhi": { state: "Delhi", lat: 28.6139, lon: 77.2090 },
  "Ahmedabad": { state: "Gujarat", lat: 23.0225, lon: 72.5714 },
  "Surat": { state: "Gujarat", lat: 21.1702, lon: 72.8311 },
  "Faridabad": { state: "Haryana", lat: 28.4089, lon: 77.3178 },
  "Gurgaon": { state: "Haryana", lat: 28.4595, lon: 77.0266 },
  "Jamshedpur": { state: "Jharkhand", lat: 22.8046, lon: 86.2029 },
  "Ranchi": { state: "Jharkhand", lat: 23.3441, lon: 85.3096 },
  "Bangalore": { state: "Karnataka", lat: 12.9716, lon: 77.5946 },
  "Mangalore": { state: "Karnataka", lat: 12.9141, lon: 74.8560 },
  "Mysore": { state: "Karnataka", lat: 12.2958, lon: 76.6394 },
  "Kochi": { state: "Kerala", lat: 9.9312, lon: 76.2673 },
  "Trivandrum": { state: "Kerala", lat: 8.5241, lon: 76.9366 },
  "Bhopal": { state: "Madhya Pradesh", lat: 23.2599, lon: 77.4126 },
  "Indore": { state: "Madhya Pradesh", lat: 22.7196, lon: 75.8577 },
  "Mumbai": { state: "Maharashtra", lat: 19.0760, lon: 72.8777 },
  "Nagpur": { state: "Maharashtra", lat: 21.1458, lon: 79.0882 },
  "Pune": { state: "Maharashtra", lat: 18.5204, lon: 73.8567 },
  "Bhubaneswar": { state: "Odisha", lat: 20.2961, lon: 85.8245 },
  "Cuttack": { state: "Odisha", lat: 20.4625, lon: 85.8828 },
  "Amritsar": { state: "Punjab", lat: 31.6340, lon: 74.8723 },
  "Ludhiana": { state: "Punjab", lat: 30.9010, lon: 75.8573 },
  "Jaipur": { state: "Rajasthan", lat: 26.9124, lon: 75.7873 },
  "Jodhpur": { state: "Rajasthan", lat: 26.2389, lon: 73.0243 },
  "Chennai": { state: "Tamil Nadu", lat: 13.0827, lon: 80.2707 },
  "Coimbatore": { state: "Tamil Nadu", lat: 11.0168, lon: 76.9558 },
  "Hyderabad": { state: "Telangana", lat: 17.3850, lon: 78.4867 },
  "Warangal": { state: "Telangana", lat: 17.9689, lon: 79.5941 },
  "Lucknow": { state: "Uttar Pradesh", lat: 26.8467, lon: 80.9462 },
  "Noida": { state: "Uttar Pradesh", lat: 28.5355, lon: 77.3910 },
  "Dehradun": { state: "Uttarakhand", lat: 30.3165, lon: 78.0322 },
  "Haridwar": { state: "Uttarakhand", lat: 29.9457, lon: 78.1642 },
  "Durgapur": { state: "West Bengal", lat: 23.5204, lon: 87.3119 },
  "Kolkata": { state: "West Bengal", lat: 22.5726, lon: 88.3639 }
};

// State Landmark Metadata Mapping
const STATE_METADATA: Record<string, { image: string; landmark: string; tagline: string }> = {
  "Rajasthan": {
    image: "https://images.unsplash.com/photo-1599661046289-e31897846e41?auto=format&fit=crop&w=1200&q=80",
    landmark: "✨ Hawa Mahal & Pink City",
    tagline: "Land of Forts, Royal Heritage & Rapidly Growing Housing Hubs"
  },
  "Maharashtra": {
    image: "https://images.unsplash.com/photo-1570168007204-dfb528c6958f?auto=format&fit=crop&w=1200&q=80",
    landmark: "✨ Gateway of India & Coastal Heights",
    tagline: "India's Commercial Capital & Premium Coastal Real Estate"
  },
  "Karnataka": {
    image: "https://images.unsplash.com/photo-1596176530529-78163a4f7af2?auto=format&fit=crop&w=1200&q=80",
    landmark: "✨ Bangalore Tech Parks & Palaces",
    tagline: "Silicon Valley of India & High-Growth IT Real Estate Corridor"
  },
  "Delhi": {
    image: "https://images.unsplash.com/photo-1587474260584-136574528ed5?auto=format&fit=crop&w=1200&q=80",
    landmark: "✨ India Gate & Capital Skyline",
    tagline: "National Capital Region with Luxury Highrises & Metro Connectivity"
  },
  "Tamil Nadu": {
    image: "https://images.unsplash.com/photo-1582510003544-4d00b7f74220?auto=format&fit=crop&w=1200&q=80",
    landmark: "✨ Dravidian Heritage & Chennai Coast",
    tagline: "Industrial Titan & Automobile Capital Real Estate"
  },
  "Telangana": {
    image: "https://images.unsplash.com/photo-1605649487212-47bdab064df7?auto=format&fit=crop&w=1200&q=80",
    landmark: "✨ Charminar & Cyberabad IT District",
    tagline: "Biotech & Cyber Corridor with Premium Gated Communities"
  },
  "West Bengal": {
    image: "https://images.unsplash.com/photo-1558431382-27e303142255?auto=format&fit=crop&w=1200&q=80",
    landmark: "✨ Howrah Bridge & Cultural Heritage",
    tagline: "Cultural Capital & Emerging Eastern Financial Hub"
  }
};

const DEFAULT_METADATA = {
  image: "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&fit=crop&w=1200&q=80",
  landmark: "✨ Premium Architecture",
  tagline: "Exploring High-Growth Property Markets Across India"
};

let serverMetadata: MetadataResponse | null = null;
let analyticsData: AnalyticsResponse | null = null;
let leafletMap: L.Map | null = null;
let cityMarkers: Record<string, L.Marker> = {};
let currentTileLayer: L.TileLayer | null = null;

// Tile Layer URLs
const TILE_LAYERS = {
  satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  streets: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initGSAPAnimations();
  initLeafletMap();
  fetchServerData();
  setupNavigationEvents();
  setupFormEvents();
});

// GSAP Entrance Animations
function initGSAPAnimations(): void {
  gsap.from('.gsap-hero', {
    opacity: 0,
    y: -30,
    duration: 1.0,
    ease: 'power3.out'
  });

  gsap.from('.gsap-location', {
    opacity: 0,
    y: 35,
    duration: 0.9,
    delay: 0.2,
    ease: 'power3.out'
  });

  gsap.from('.gsap-form', {
    opacity: 0,
    y: 40,
    duration: 0.9,
    delay: 0.4,
    ease: 'power3.out'
  });
}

// Leaflet Map Initialization
function initLeafletMap(): void {
  const mapContainer = document.getElementById('leaflet-map');
  if (!mapContainer) return;

  // Centered on India
  leafletMap = L.map('leaflet-map', {
    center: [22.5, 78.5],
    zoom: 5,
    zoomControl: true
  });

  // Default Satellite Tile Layer
  currentTileLayer = L.tileLayer(TILE_LAYERS.satellite, {
    maxZoom: 18,
    attribution: '© Esri — Satellite imagery'
  }).addTo(leafletMap);

  setupMapLayerToggles();
}

// Map Layer Style Switcher
function setupMapLayerToggles(): void {
  const btnSat = document.getElementById('map-style-sat');
  const btnStreet = document.getElementById('map-style-street');
  const btnDark = document.getElementById('map-style-dark');

  const updateActiveLayer = (btn: HTMLElement | null, tileUrl: string, attr: string) => {
    [btnSat, btnStreet, btnDark].forEach(b => b?.classList.remove('active'));
    btn?.classList.add('active');

    if (leafletMap && currentTileLayer) {
      leafletMap.removeLayer(currentTileLayer);
      currentTileLayer = L.tileLayer(tileUrl, { maxZoom: 18, attribution: attr }).addTo(leafletMap);
    }
  };

  btnSat?.addEventListener('click', () => updateActiveLayer(btnSat, TILE_LAYERS.satellite, '© Esri Satellite'));
  btnStreet?.addEventListener('click', () => updateActiveLayer(btnStreet, TILE_LAYERS.streets, '© OpenStreetMap'));
  btnDark?.addEventListener('click', () => updateActiveLayer(btnDark, TILE_LAYERS.dark, '© CARTO Dark'));
}

// Fetch Server Metadata & Analytics Data
async function fetchServerData(): Promise<void> {
  try {
    const [metaRes, analyticsRes] = await Promise.all([
      fetch('/api/metadata'),
      fetch('/api/analytics')
    ]);

    if (metaRes.ok) {
      serverMetadata = await metaRes.json();
      populateLocationDropdowns();
    }
    if (analyticsRes.ok) {
      analyticsData = await analyticsRes.json();
      populateMapMarkers();
      renderAnalyticsCharts();
    }
  } catch (err) {
    console.warn("Could not fetch server data, initializing fallback:", err);
  }
}

// Populate Map Markers for 42 Cities
function populateMapMarkers(): void {
  if (!leafletMap) return;

  const cityStats = analyticsData?.city_stats;
  
  // Custom Red Icon for Active City
  const customIcon = L.divIcon({
    className: 'custom-map-pin',
    html: `<div style="background-color: #f43f5e; width: 14px; height: 14px; border-radius: 50%; border: 2px solid #ffffff; box-shadow: 0 0 10px #f43f5e;"></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7]
  });

  for (const [cityName, info] of Object.entries(CITY_COORDINATES)) {
    const stats = cityStats ? cityStats[cityName] : null;
    const avgPriceText = stats ? `₹ ${stats.avg_price_lakhs} Lakhs` : 'N/A';
    const avgRateText = stats ? `₹ ${stats.avg_rate_per_sqft.toLocaleString()}/sqft` : 'N/A';

    const popupContent = `
      <div style="font-family: 'Plus Jakarta Sans', sans-serif; color: #0f172a; padding: 4px;">
        <div style="font-weight: 700; font-size: 1rem; color: #0284c7;">📍 ${cityName}, ${info.state}</div>
        <div style="font-size: 0.85rem; margin-top: 4px; color: #334155;">
          <div><b>Avg Price:</b> ${avgPriceText}</div>
          <div><b>Avg Rate:</b> ${avgRateText}</div>
          ${stats ? `<div><b>Listings:</b> ${stats.total_listings.toLocaleString()}</div>` : ''}
        </div>
      </div>
    `;

    const marker = L.marker([info.lat, info.lon], { icon: customIcon })
      .addTo(leafletMap)
      .bindPopup(popupContent);

    // Click marker selects city & state in app!
    marker.on('click', () => {
      selectLocationFromMap(info.state, cityName);
    });

    cityMarkers[cityName] = marker;
  }
}

// Map Marker Click Handler
function selectLocationFromMap(state: string, city: string): void {
  const stateSelect = document.getElementById('state-select') as HTMLSelectElement;
  const citySelect = document.getElementById('city-select') as HTMLSelectElement;

  if (stateSelect && stateSelect.value !== state) {
    stateSelect.value = state;
    updateCitiesForState(state);
  }

  if (citySelect) {
    citySelect.value = city;
    updateLocalitiesForCity(city);
  }

  updateCityBenchmark(city);
  updateStateBanner(state);
}

// Populate Location Selectors
function populateLocationDropdowns(): void {
  if (!serverMetadata) return;

  const stateSelect = document.getElementById('state-select') as HTMLSelectElement;
  if (stateSelect && serverMetadata.states) {
    stateSelect.innerHTML = '';
    serverMetadata.states.forEach((st) => {
      const opt = document.createElement('option');
      opt.value = st;
      opt.textContent = st;
      stateSelect.appendChild(opt);
    });

    if (serverMetadata.states.includes("Rajasthan")) {
      stateSelect.value = "Rajasthan";
    }
    updateCitiesForState(stateSelect.value);
  }
}

function updateCitiesForState(state: string): void {
  const citySelect = document.getElementById('city-select') as HTMLSelectElement;
  if (!citySelect) return;

  citySelect.innerHTML = '';

  const cities = serverMetadata?.cities_by_state[state] || ["Jaipur", "Jodhpur"];
  cities.forEach((ct) => {
    const opt = document.createElement('option');
    opt.value = ct;
    opt.textContent = ct;
    citySelect.appendChild(opt);
  });

  if (cities.length > 0) {
    citySelect.value = cities[0];
    updateLocalitiesForCity(cities[0]);
    updateCityBenchmark(cities[0]);
  }

  updateStateBanner(state);
}

function updateLocalitiesForCity(city: string): void {
  const localitySelect = document.getElementById('locality-select') as HTMLSelectElement;
  if (!localitySelect) return;

  localitySelect.innerHTML = '';

  const localities = serverMetadata?.localities_by_city[city] || ["Locality_1", "Locality_2"];
  localities.forEach((loc) => {
    const opt = document.createElement('option');
    opt.value = loc;
    opt.textContent = loc;
    localitySelect.appendChild(opt);
  });

  updateCityBenchmark(city);
}

// Update City Benchmark Box
function updateCityBenchmark(city: string): void {
  const nameEl = document.getElementById('avg-city-name');
  const priceEl = document.getElementById('avg-city-price');
  const rateEl = document.getElementById('avg-city-rate');

  if (!analyticsData?.city_stats[city]) return;
  const stats = analyticsData.city_stats[city];

  if (nameEl) nameEl.textContent = city;
  if (priceEl) priceEl.textContent = `₹ ${stats.avg_price_lakhs} Lakhs`;
  if (rateEl) rateEl.textContent = `₹ ${stats.avg_rate_per_sqft.toLocaleString()}/sqft`;

  // Pan Leaflet Map to City Coordinates
  if (leafletMap && stats.lat && stats.lon) {
    leafletMap.panTo([stats.lat, stats.lon], { animate: true, duration: 1.0 });
  }
}

// Update Visual State Landmark Banner
function updateStateBanner(state: string): void {
  const bannerCard = document.getElementById('state-banner-card');
  const landmarkTag = document.getElementById('state-landmark-tag');
  const bannerTitle = document.getElementById('state-banner-title');
  const bannerDesc = document.getElementById('state-banner-desc');

  if (!bannerCard || !landmarkTag || !bannerTitle || !bannerDesc) return;

  const meta = STATE_METADATA[state] || DEFAULT_METADATA;

  gsap.to(bannerCard, {
    opacity: 0.4,
    duration: 0.2,
    onComplete: () => {
      bannerCard.style.backgroundImage = `url('${meta.image}')`;
      landmarkTag.textContent = meta.landmark;
      bannerTitle.textContent = `${state} Real Estate Market`;
      bannerDesc.textContent = meta.tagline;

      gsap.to(bannerCard, { opacity: 1.0, duration: 0.35, ease: 'power2.out' });
    }
  });
}

// Navigation & Three Dots Toggle Setup
function setupNavigationEvents(): void {
  const btnValuation = document.getElementById('nav-valuation-btn');
  const btnDashboard = document.getElementById('nav-dashboard-btn');
  const btnThreeDots = document.getElementById('three-dots-btn');

  const viewValuation = document.getElementById('view-valuation');
  const viewDashboard = document.getElementById('view-dashboard');

  const switchToView = (showDashboard: boolean) => {
    if (showDashboard) {
      btnValuation?.classList.remove('active');
      btnDashboard?.classList.add('active');

      gsap.to(viewValuation, {
        opacity: 0,
        duration: 0.25,
        onComplete: () => {
          viewValuation?.classList.add('hidden');
          viewDashboard?.classList.remove('hidden');
          gsap.fromTo(viewDashboard, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.4 });
        }
      });
    } else {
      btnDashboard?.classList.remove('active');
      btnValuation?.classList.add('active');

      gsap.to(viewDashboard, {
        opacity: 0,
        duration: 0.25,
        onComplete: () => {
          viewDashboard?.classList.add('hidden');
          viewValuation?.classList.remove('hidden');
          gsap.fromTo(viewValuation, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.4 });
          leafletMap?.invalidateSize();
        }
      });
    }
  };

  btnValuation?.addEventListener('click', () => switchToView(false));
  btnDashboard?.addEventListener('click', () => switchToView(true));
  
  // Three Dots button toggles dashboard
  btnThreeDots?.addEventListener('click', () => {
    const isDashboardHidden = viewDashboard?.classList.contains('hidden');
    switchToView(Boolean(isDashboardHidden));
  });
}

// Setup Form Controls
function setupFormEvents(): void {
  const stateSelect = document.getElementById('state-select') as HTMLSelectElement;
  const citySelect = document.getElementById('city-select') as HTMLSelectElement;
  const bhkInput = document.getElementById('bhk-input') as HTMLInputElement;
  const yearInput = document.getElementById('year-built-input') as HTMLInputElement;
  const form = document.getElementById('valuation-form') as HTMLFormElement;

  if (stateSelect) {
    stateSelect.addEventListener('change', (e) => {
      const selected = (e.target as HTMLSelectElement).value;
      updateCitiesForState(selected);
    });
  }

  if (citySelect) {
    citySelect.addEventListener('change', (e) => {
      const selected = (e.target as HTMLSelectElement).value;
      updateLocalitiesForCity(selected);
    });
  }

  if (bhkInput) {
    bhkInput.addEventListener('input', () => {
      const valSpan = document.getElementById('bhk-val');
      if (valSpan) valSpan.textContent = `${bhkInput.value} BHK`;
    });
  }

  if (yearInput) {
    yearInput.addEventListener('input', () => {
      const valSpan = document.getElementById('year-val');
      if (valSpan) valSpan.textContent = yearInput.value;
    });
  }

  if (form) {
    form.addEventListener('submit', handleFormSubmit);
  }
}

// Handle Prediction Form Submission
async function handleFormSubmit(e: Event): Promise<void> {
  e.preventDefault();

  const state = (document.getElementById('state-select') as HTMLSelectElement).value;
  const city = (document.getElementById('city-select') as HTMLSelectElement).value;
  const locality = (document.getElementById('locality-select') as HTMLSelectElement).value;
  const propertyType = (document.getElementById('property-type-select') as HTMLSelectElement).value;
  const bhk = parseInt((document.getElementById('bhk-input') as HTMLInputElement).value, 10);
  const sizeSqft = parseInt((document.getElementById('size-input') as HTMLInputElement).value, 10);
  const floorNo = parseInt((document.getElementById('floor-no-input') as HTMLInputElement).value, 10);
  const totalFloors = parseInt((document.getElementById('total-floors-input') as HTMLInputElement).value, 10);
  const yearBuilt = parseInt((document.getElementById('year-built-input') as HTMLInputElement).value, 10);
  const ageOfProperty = Math.max(0, 2026 - yearBuilt);

  const furnishing = (document.getElementById('furnishing-select') as HTMLSelectElement).value;
  const facing = (document.getElementById('facing-select') as HTMLSelectElement).value;
  const ownerType = (document.getElementById('owner-type-select') as HTMLSelectElement).value;
  const availability = (document.getElementById('availability-select') as HTMLSelectElement).value;
  const parking = (document.getElementById('parking-select') as HTMLSelectElement).value;
  const security = (document.getElementById('security-select') as HTMLSelectElement).value;
  const transport = (document.getElementById('transport-select') as HTMLSelectElement).value;

  const amenitiesList: string[] = [];
  if ((document.getElementById('am-gym') as HTMLInputElement).checked) amenitiesList.push("Gym");
  if ((document.getElementById('am-pool') as HTMLInputElement).checked) amenitiesList.push("Pool");
  if ((document.getElementById('am-garden') as HTMLInputElement).checked) amenitiesList.push("Garden");
  if ((document.getElementById('am-playground') as HTMLInputElement).checked) amenitiesList.push("Playground");
  if ((document.getElementById('am-clubhouse') as HTMLInputElement).checked) amenitiesList.push("Clubhouse");
  const amenitiesStr = amenitiesList.length > 0 ? amenitiesList.join(", ") : "None";

  const payload = {
    State: state,
    City: city,
    Locality: locality,
    Property_Type: propertyType,
    BHK: bhk,
    Size_in_SqFt: sizeSqft,
    Year_Built: yearBuilt,
    Furnished_Status: furnishing,
    Floor_No: floorNo,
    Total_Floors: totalFloors,
    Age_of_Property: ageOfProperty,
    Nearby_Schools: 5,
    Nearby_Hospitals: 4,
    Public_Transport_Accessibility: transport,
    Parking_Space: parking,
    Security: security,
    Amenities: amenitiesStr,
    Facing: facing,
    Owner_Type: ownerType,
    Availability_Status: availability
  };

  const submitBtn = document.getElementById('submit-btn') as HTMLButtonElement;
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>⚡ Calculating Valuation...</span>';
  }

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data: PredictionResponse = await res.json();
      displayResults(data, payload);
    } else {
      alert("Failed to compute prediction. Ensure python server.py is running!");
    }
  } catch (err) {
    console.error("Prediction API call error:", err);
    alert("Connection error: Make sure the Python backend server (server.py) is running on port 8000.");
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<span>💰 Calculate Price Valuation</span>';
    }
  }
}

// Display Prediction Results with GSAP Counter
function displayResults(data: PredictionResponse, payload: any): void {
  const resultSection = document.getElementById('results-section');
  const mainCounter = document.getElementById('price-main-counter');
  const subCounter = document.getElementById('price-sub-counter');
  const rangeVal = document.getElementById('price-range-val');
  const resBhkType = document.getElementById('res-bhk-type');
  const resArea = document.getElementById('res-area');
  const resRate = document.getElementById('res-rate');
  const resLocation = document.getElementById('res-location');

  if (!resultSection || !mainCounter || !subCounter) return;

  resultSection.classList.remove('hidden');

  if (rangeVal && data.price_range) {
    rangeVal.textContent = data.price_range;
  }

  if (resBhkType) resBhkType.textContent = `${payload.BHK} BHK ${payload.Property_Type}`;
  if (resArea) resArea.textContent = `${payload.Size_in_SqFt.toLocaleString()} sq ft`;
  if (resRate) resRate.textContent = `₹ ${data.rate_per_sqft.toLocaleString()}/sqft`;
  if (resLocation) resLocation.textContent = `${payload.City}, ${payload.State}`;

  const counterObj = { lakhs: 0, crores: 0 };

  gsap.to(counterObj, {
    lakhs: data.price_lakhs,
    crores: data.price_crores,
    duration: 1.5,
    ease: 'power2.out',
    onUpdate: () => {
      if (data.price_lakhs >= 100) {
        mainCounter.textContent = `₹ ${counterObj.crores.toFixed(2)} Cr`;
        subCounter.textContent = `₹ ${counterObj.lakhs.toLocaleString('en-IN', { maximumFractionDigits: 2 })} Lakhs`;
      } else {
        mainCounter.textContent = `₹ ${counterObj.lakhs.toLocaleString('en-IN', { maximumFractionDigits: 2 })} Lakhs`;
        subCounter.textContent = `₹ ${(counterObj.crores).toFixed(4)} Crores`;
      }
    }
  });

  setupExportAndComparator();
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Setup Print Export & City Comparator Logic
function setupExportAndComparator(): void {
  const exportBtn = document.getElementById('export-pdf-btn');
  exportBtn?.addEventListener('click', () => {
    window.print();
  });

  const cmp1 = document.getElementById('cmp-city-1') as HTMLSelectElement;
  const cmp2 = document.getElementById('cmp-city-2') as HTMLSelectElement;

  const updateCmp = () => {
    if (!analyticsData?.city_stats) return;
    const c1 = cmp1?.value || "Mumbai";
    const c2 = cmp2?.value || "Jaipur";

    const s1 = analyticsData.city_stats[c1];
    const s2 = analyticsData.city_stats[c2];

    const box1Title = document.querySelector('#cmp-box-1 .cmp-city-title');
    const box1Price = document.getElementById('cmp-price-1');
    const box1Rate = document.getElementById('cmp-rate-1');

    if (box1Title && s1) box1Title.textContent = `${c1}, ${s1.state}`;
    if (box1Price && s1) box1Price.textContent = `₹ ${s1.avg_price_lakhs} Lakhs`;
    if (box1Rate && s1) box1Rate.textContent = `₹ ${s1.avg_rate_per_sqft.toLocaleString()}/sqft`;

    const box2Title = document.querySelector('#cmp-box-2 .cmp-city-title');
    const box2Price = document.getElementById('cmp-price-2');
    const box2Rate = document.getElementById('cmp-rate-2');

    if (box2Title && s2) box2Title.textContent = `${c2}, ${s2.state}`;
    if (box2Price && s2) box2Price.textContent = `₹ ${s2.avg_price_lakhs} Lakhs`;
    if (box2Rate && s2) box2Rate.textContent = `₹ ${s2.avg_rate_per_sqft.toLocaleString()}/sqft`;
  };

  cmp1?.addEventListener('change', updateCmp);
  cmp2?.addEventListener('change', updateCmp);
  updateCmp();
}


// Render Chart.js Market Analytics Dashboard
function renderAnalyticsCharts(): void {
  if (!analyticsData) return;

  // 1. State Rate Comparison Bar Chart
  const stateCanvas = document.getElementById('chart-state-rate') as HTMLCanvasElement;
  if (stateCanvas) {
    const statesList = Object.keys(analyticsData.state_stats).slice(0, 10);
    const stateRates = statesList.map(st => analyticsData!.state_stats[st].avg_rate_per_sqft);

    new Chart(stateCanvas, {
      type: 'bar',
      data: {
        labels: statesList,
        datasets: [{
          label: 'Avg Rate (₹/sqft)',
          data: stateRates,
          backgroundColor: '#38bdf8',
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.08)' } }
        }
      }
    });
  }

  // 2. BHK Price Trend Chart
  const bhkCanvas = document.getElementById('chart-bhk-price') as HTMLCanvasElement;
  if (bhkCanvas) {
    const bhks = Object.keys(analyticsData.bhk_stats).map(b => `${b} BHK`);
    const bhkPrices = Object.values(analyticsData.bhk_stats);

    new Chart(bhkCanvas, {
      type: 'line',
      data: {
        labels: bhks,
        datasets: [{
          label: 'Avg Price (Lakhs)',
          data: bhkPrices,
          borderColor: '#10b981',
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
          y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.08)' } }
        }
      }
    });
  }

  // 3. Property Type Doughnut Chart
  const propCanvas = document.getElementById('chart-prop-type') as HTMLCanvasElement;
  if (propCanvas) {
    const ptypes = Object.keys(analyticsData.prop_type_stats);
    const pprices = Object.values(analyticsData.prop_type_stats);

    new Chart(propCanvas, {
      type: 'doughnut',
      data: {
        labels: ptypes,
        datasets: [{
          data: pprices,
          backgroundColor: ['#38bdf8', '#818cf8', '#f43f5e']
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#f8fafc' } } }
      }
    });
  }

  // 4. Populate State Comparison Table
  const tableBody = document.getElementById('state-table-body');
  if (tableBody) {
    tableBody.innerHTML = '';
    for (const [stName, info] of Object.entries(analyticsData.state_stats)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><b>${stName}</b></td>
        <td>₹ ${info.avg_price_lakhs} Lakhs</td>
        <td>₹ ${info.avg_rate_per_sqft.toLocaleString()}/sqft</td>
        <td>${info.total_listings.toLocaleString()}</td>
      `;
      tableBody.appendChild(tr);
    }
  }
}
