<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis minScale="1e+8" version="3.2.3-Bonn" maxScale="0" hasScaleBasedVisibilityFlag="0">
  <pipe>
    <rasterrenderer type="paletted" alphaBand="-1" band="1" opacity="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry color="#8a8a8a" value="1" alpha="255" label="Revêtement artificiel au sol"/>
        <paletteEntry color="#dc0f0f" value="2" alpha="255" label="Constructions artificielles hors sol"/>
        <paletteEntry color="#ff5500" value="62" alpha="255" label="Serres"/>
        <paletteEntry color="#4e4e4e" value="3" alpha="255" label="Réseau ferroviaire"/>
        <paletteEntry color="#d0d0d0" value="4" alpha="255" label="Sols nus"/>
        <paletteEntry color="#2461f7" value="5" alpha="255" label="Eaux de surface"/>
        <paletteEntry color="#ffff73" value="6" alpha="255" label="Couvert herbacé en rotation dans l'année (ex: culture annuelle)"/>
        <paletteEntry color="#e9ffbe" value="7" alpha="255" label="Couvert herbacé toute l'année"/>
        <paletteEntry color="#003200" value="8" alpha="255" label="Résineux (> 3m)"/>
        <paletteEntry color="#007800" value="80" alpha="255" label="Résineux (≤ 3m)"/>
        <paletteEntry color="#28c828" value="9" alpha="255" label="Feuillus (> 3m)"/>
        <paletteEntry color="#b7e8b0" value="90" alpha="255" label="Feuillus (≤ 3m)"/>
        <paletteEntry color="#e5ea3f" value="0" alpha="0" label="Pas de données"/>
        <paletteEntry color="#8a8a8a" value="11" alpha="255" label="Revêtement artificiel au sol(pont)"/>
        <paletteEntry color="#8a8a8a" value="15" alpha="255" label="Revêtement artificiel au sol (sous eau)"/>
        <paletteEntry color="#8a8a8a" value="81" alpha="255" label="Résineux (sous pont)"/>
        <paletteEntry color="#8a8a8a" value="18" alpha="255" label="Revêtement artificiel au sol (sous résineux)"/>
        <paletteEntry color="#8a8a8a" value="31" alpha="255" label="Revêtement artificiel au sol (pont sur réseau ferroviaire)"/>
        <paletteEntry color="#8a8a8a" value="71" alpha="255" label="Couvert herbacé toute l'année (sous pont)"/>
        <paletteEntry color="#8a8a8a" value="51" alpha="255" label="Ponts sur l'eau"/>
        <paletteEntry color="#8a8a8a" value="91" alpha="255" label="Feuillus (sous pont)"/>
        <paletteEntry color="#8a8a8a" value="19" alpha="255" label="Revêtement artificiel au sol (sous feuillus)"/>
        <paletteEntry color="#dc0f0f" value="28" alpha="255" label="Constr. artificielles hors sol (sous résineux)"/>
        <paletteEntry color="#dc0f0f" value="29" alpha="255" label="Constr. artificielles hors sol (sous feuillus)"/>
        <paletteEntry color="#4e4e4e" value="38" alpha="255" label="Réseau ferroviaire (sous résineux)"/>
        <paletteEntry color="#4e4e4e" value="39" alpha="255" label="Réseau ferroviaire (sous feuillus)"/>
        <paletteEntry color="#4e4e4e" value="93" alpha="255" label="Feuillus (sous réseau ferroviaire)"/>
        <paletteEntry color="#2461f7" value="55" alpha="255" label="Eaux de surface (2 niveaux)"/>
        <paletteEntry color="#2461f7" value="58" alpha="255" label="Eaux de surface (sous résineux)"/>
        <paletteEntry color="#2461f7" value="75" alpha="255" label="Couvert herbacé toute l'année (sous canal)"/>
        <paletteEntry color="#2461f7" value="59" alpha="255" label="Eaux de surface (sous feuillus)"/>
      </colorPalette>
      <colorramp type="randomcolors" name="[source]"/>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0"/>
    <huesaturation saturation="0" colorizeOn="0" colorizeBlue="128" colorizeRed="255" colorizeStrength="100" grayscaleMode="0" colorizeGreen="128"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
