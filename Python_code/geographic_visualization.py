"""
SDS6210_InformaticsForHealth - Geographic Visualization
Part VII: County-level Choropleth Maps

Author: Cavin Otieno
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

print("=" * 60)
print("GEOGRAPHIC VISUALIZATION")
print("=" * 60)

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Load data
df = pd.read_csv('data/patient_survival_data.csv')

# Create simulated county data for Kenyan counties
# This demonstrates geographic visualization with synthetic data
kenyan_counties = [
    'Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika',
    'Malindi', 'Kitale', 'Garissa', 'Meru', 'Nyeri', 'Kakamega',
    'Kericho', 'Embu', 'Migori', 'Homa Bay', 'Siaya', 'Bungoma',
    'Baringo', 'Laikipia', 'Machakos', 'Kajiado', 'Keroka', 'Moyale'
]

# Assign patients to counties with simulated geographic patterns
np.random.seed(6210)

# Create geographic clusters based on cardiovascular risk
# Urban areas (Nairobi, Mombasa, Kisumu) tend to have higher risk
urban_counties = ['Nairobi', 'Mombasa', 'Kisumu']
highland_counties = ['Nakuru', 'Eldoret', 'Kericho', 'Laikipia']
coastal_counties = ['Malindi', 'Garissa', 'Meru']

# Generate county assignments with geographic patterns
county_assignments = []
for i in range(len(df)):
    if df.loc[i, 'age'] > 55 and df.loc[i, 'sbp_mmHg'] > 135:
        # Older patients with high BP more likely in urban areas
        county = np.random.choice(urban_counties + highland_counties, 
                                  p=[0.4, 0.3, 0.15, 0.15])
    else:
        county = np.random.choice(kenyan_counties)
    county_assignments.append(county)

df['county'] = county_assignments

# Aggregate data by county
county_stats = df.groupby('county').agg({
    'ascvd_event': ['sum', 'count', 'mean'],
    'sbp_mmHg': 'mean',
    'diabetes': 'mean',
    'age': 'mean'
}).round(2)

county_stats.columns = ['events', 'total_patients', 'event_rate', 'mean_sbp', 'diabetes_rate', 'mean_age']
county_stats = county_stats.reset_index()

# Create simulated geographic coordinates for visualization
# (Longitude, Latitude) for major Kenyan counties
county_coords = {
    'Nairobi': (36.82, -1.29), 'Mombasa': (39.66, -4.04), 'Kisumu': (35.29, -0.10),
    'Nakuru': (36.07, -0.30), 'Eldoret': (35.28, 0.51), 'Thika': (37.07, -1.03),
    'Malindi': (40.13, -3.21), 'Kitale': (35.00, 1.02), 'Garissa': (39.64, -0.45),
    'Meru': (37.65, 0.05), 'Nyeri': (36.96, -0.42), 'Kakamega': (34.75, 0.28),
    'Kericho': (35.29, -0.37), 'Embu': (37.46, -0.53), 'Migori': (34.47, -1.07),
    'Homa Bay': (34.45, -0.53), 'Siaya': (34.26, -0.06), 'Bungoma': (34.56, 0.56),
    'Baringo': (35.97, 0.47), 'Laikipia': (36.78, 0.30), 'Machakos': (37.26, -1.52),
    'Kajiado': (36.78, -1.85), 'Keroka': (34.90, -0.77), 'Moyale': (39.06, 3.53)
}

# Add coordinates to county_stats
county_stats['lon'] = county_stats['county'].map(lambda x: county_coords.get(x, (0, 0))[0])
county_stats['lat'] = county_stats['county'].map(lambda x: county_coords.get(x, (0, 0))[1])

# Save county-level data
county_stats.to_csv('data/county_level_statistics.csv', index=False)
print("\nCounty-level statistics saved to data/county_level_statistics.csv")

# Create Visualization 1: Event Rate Choropleth Map
fig, ax = plt.subplots(figsize=(12, 10))

# Normalize event rates for color mapping
norm = Normalize(vmin=county_stats['event_rate'].min(), vmax=county_stats['event_rate'].max())
cmap = plt.cm.RdYlGn_r  # Red for high risk, Green for low risk

# Plot counties as scatter points with size based on patient count
sizes = county_stats['total_patients'] * 5
colors = county_stats['event_rate']

scatter = ax.scatter(county_stats['lon'], county_stats['lat'], 
                     c=colors, s=sizes, cmap=cmap, norm=norm,
                     alpha=0.7, edgecolors='black', linewidth=0.5)

# Add county labels
for idx, row in county_stats.iterrows():
    ax.annotate(row['county'], (row['lon'], row['lat']), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
cbar.set_label('ASCVD Event Rate', fontsize=12)

# Set axis labels and title
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Kenya: ASCVD Event Rate by County\n(Bubble size = Number of patients)', 
             fontsize=14, fontweight='bold')

# Add Kenya outline approximation
ax.set_xlim(33.5, 42)
ax.set_ylim(-5, 5.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/choropleth_event_rate.png', dpi=150, bbox_inches='tight')
print("Saved: visualizations/choropleth_event_rate.png")
plt.close()

# Create Visualization 2: SBP Choropleth Map
fig, ax = plt.subplots(figsize=(12, 10))

norm2 = Normalize(vmin=county_stats['mean_sbp'].min(), vmax=county_stats['mean_sbp'].max())
cmap2 = plt.cm.YlOrRd  # Yellow-Orange-Red for blood pressure

sizes2 = county_stats['total_patients'] * 5
colors2 = county_stats['mean_sbp']

scatter2 = ax.scatter(county_stats['lon'], county_stats['lat'], 
                      c=colors2, s=sizes2, cmap=cmap2, norm=norm2,
                      alpha=0.7, edgecolors='black', linewidth=0.5)

for idx, row in county_stats.iterrows():
    ax.annotate(row['county'], (row['lon'], row['lat']), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold')

cbar2 = plt.colorbar(ScalarMappable(norm=norm2, cmap=cmap2), ax=ax)
cbar2.set_label('Mean SBP (mmHg)', fontsize=12)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Kenya: Mean Systolic Blood Pressure by County\n(Bubble size = Number of patients)', 
             fontsize=14, fontweight='bold')

ax.set_xlim(33.5, 42)
ax.set_ylim(-5, 5.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/choropleth_sbp.png', dpi=150, bbox_inches='tight')
print("Saved: visualizations/choropleth_sbp.png")
plt.close()

# Create Visualization 3: Dual Variable Map (Event Rate + Diabetes)
fig, ax = plt.subplots(figsize=(12, 10))

# Create combined risk score
county_stats['risk_score'] = (county_stats['event_rate'] * 100 + 
                              county_stats['diabetes_rate'] * 50 + 
                              (county_stats['mean_sbp'] - 120) / 10)

norm3 = Normalize(vmin=county_stats['risk_score'].min(), vmax=county_stats['risk_score'].max())
cmap3 = plt.cm.plasma  # Plasma colormap for risk

sizes3 = county_stats['total_patients'] * 5
colors3 = county_stats['risk_score']

scatter3 = ax.scatter(county_stats['lon'], county_stats['lat'], 
                      c=colors3, s=sizes3, cmap=cmap3, norm=norm3,
                      alpha=0.7, edgecolors='black', linewidth=0.5)

for idx, row in county_stats.iterrows():
    ax.annotate(row['county'], (row['lon'], row['lat']), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold')

cbar3 = plt.colorbar(ScalarMappable(norm=norm3, cmap=cmap3), ax=ax)
cbar3.set_label('Composite Risk Score', fontsize=12)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Kenya: Composite Cardiovascular Risk Score by County\n(Event Rate + Diabetes + SBP)', 
             fontsize=14, fontweight='bold')

ax.set_xlim(33.5, 42)
ax.set_ylim(-5, 5.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/choropleth_risk_score.png', dpi=150, bbox_inches='tight')
print("Saved: visualizations/choropleth_risk_score.png")
plt.close()

# Print County Statistics Summary
print("\n" + "=" * 60)
print("COUNTY-LEVEL STATISTICS SUMMARY")
print("=" * 60)
print(f"\nTotal Counties Analyzed: {len(county_stats)}")
print(f"Total Patients: {county_stats['total_patients'].sum()}")
print(f"Total Events: {county_stats['events'].sum()}")
print(f"\nOverall Event Rate: {county_stats['events'].sum() / county_stats['total_patients'].sum():.2%}")

print("\nTop 5 Counties by Event Rate:")
top_counties = county_stats.nlargest(5, 'event_rate')[['county', 'event_rate', 'total_patients', 'mean_sbp']]
print(top_counties.to_string(index=False))

print("\nTop 5 Counties by Composite Risk Score:")
top_risk = county_stats.nlargest(5, 'risk_score')[['county', 'risk_score', 'event_rate', 'diabetes_rate']]
print(top_risk.to_string(index=False))

# Create summary bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart 1: Event rates by county
sorted_by_event = county_stats.sort_values('event_rate', ascending=True)
colors_bar1 = plt.cm.RdYlGn_r(norm(sorted_by_event['event_rate']))

axes[0].barh(sorted_by_event['county'], sorted_by_event['event_rate'], color=colors_bar1)
axes[0].set_xlabel('ASCVD Event Rate', fontsize=12)
axes[0].set_title('Event Rate by County', fontsize=12, fontweight='bold')
axes[0].axvline(x=county_stats['event_rate'].mean(), color='black', linestyle='--', 
                label=f'Mean: {county_stats["event_rate"].mean():.2%}')
axes[0].legend()

# Bar chart 2: Patient distribution
sorted_by_patients = county_stats.sort_values('total_patients', ascending=True)
colors_bar2 = plt.cm.Blues(np.linspace(0.3, 0.9, len(sorted_by_patients)))

axes[1].barh(sorted_by_patients['county'], sorted_by_patients['total_patients'], color=colors_bar2)
axes[1].set_xlabel('Number of Patients', fontsize=12)
axes[1].set_title('Patient Distribution by County', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/county_bar_charts.png', dpi=150, bbox_inches='tight')
print("\nSaved: visualizations/county_bar_charts.png")
plt.close()

print("\n" + "=" * 60)
print("GEOGRAPHIC VISUALIZATION COMPLETE!")
print("=" * 60)
print("\nGenerated Visualizations:")
print("  1. visualizations/choropleth_event_rate.png")
print("  2. visualizations/choropleth_sbp.png")
print("  3. visualizations/choropleth_risk_score.png")
print("  4. visualizations/county_bar_charts.png")
print("\nData saved to: data/county_level_statistics.csv")
