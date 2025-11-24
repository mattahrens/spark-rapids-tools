/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.tool.planparser

import java.nio.file.Paths

import scala.util.matching.Regex

import org.json4s.{DefaultFormats, Formats}
import org.json4s.jackson.JsonMethods

import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.tool.util.UTF8Source

// Utilities used to handle Gluten Ops
object GlutenParseHelper extends Logging {
  // Gluten operator patterns and mappings
  // Gluten uses various prefixes: ArrowFile, Arrow, Columnar, Velox, CH, and Transformer suffix
  private val GLUTEN_PATTERN: Regex = (
    "(ArrowFile|Arrow|Columnar|Velox|CH|RowToVeloxColumnar|RowToCHColumnar|" +
    ".*Transformer|.*Columnar|HashAggregateTransformer|ShuffledHashJoinTransformer|" +
    "RegularHashAggregateExec|FlushableHashAggregateExec|ShuffledHashJoinExec|" +
    "InputIterator|ScanTransformer)"
  ).r
  private val GLUTEN_OPS_MAPPING_DIR = "glutenOperatorMappings"
  private val DEFAULT_GLUTEN_OPS_MAPPING_FILE = "gluten-default.json"

  /**
   * Checks if the properties indicate that the application is a Gluten app.
   * This can be checked by looking for Gluten-specific properties.
   *
   * @param properties spark properties captured from the eventlog environment details
   * @return true if the properties indicate that it is a Gluten app.
   */
  def isGlutenApp(properties: collection.Map[String, String]): Boolean = {
    // Check if Gluten plugin is enabled
    val hasPlugin = properties.get("spark.plugins")
      .exists(_.contains("org.apache.gluten.GlutenPlugin"))
    // Check if any Gluten-specific properties exist
    val hasGlutenProps = properties.keys.exists(_.startsWith("spark.gluten"))
    val isGluten = hasPlugin || hasGlutenProps

    if (isGluten) {
      logInfo(s"[GLUTEN-DEBUG] isGlutenApp: Detected Gluten app. " +
        s"Has plugin: $hasPlugin, Has gluten props: $hasGlutenProps")
      // Log some Gluten properties for debugging
      val glutenProps = properties.filterKeys(_.startsWith("spark.gluten")).take(5)
      if (glutenProps.nonEmpty) {
        logInfo(s"[GLUTEN-DEBUG] isGlutenApp: Sample Gluten properties: " +
          s"${glutenProps.keys.mkString(", ")}")
      }
    } else {
      logDebug(s"[GLUTEN-DEBUG] isGlutenApp: Not a Gluten app")
    }
    isGluten
  }

  /**
   * Maps the Gluten operator names to Spark operator names using a mapping JSON file.
   */
  private lazy val glutenToSparkMapping: Map[String, String] = {
    val mappingFile = Paths.get(GLUTEN_OPS_MAPPING_DIR, DEFAULT_GLUTEN_OPS_MAPPING_FILE).toString
    val jsonString = UTF8Source.fromResource(mappingFile).mkString
    val json = JsonMethods.parse(jsonString)
    // Implicitly define JSON formats for deserialization using DefaultFormats
    implicit val formats: Formats = DefaultFormats
    // Extract and deserialize the JValue object into a Map[String, String]
    // Currently, only the first mapping in the list is used.
    json.extract[Map[String, List[String]]]
      .iterator
      .map { case (k, v) => k -> v.head }
      .toMap
  }

  /**
   * Checks if the node name is a Gluten node.
   */
  def isGlutenNode(nodeName: String): Boolean = {
    // Check for common Gluten patterns
    // Check exact matches first for operators that might be mapped
    val trimmed = nodeName.trim
    val isGluten = trimmed == "HashAggregateTransformer" ||
    trimmed == "HashAggregateTransformerExec" ||
    trimmed == "ShuffledHashJoinTransformer" ||
    trimmed == "ShuffledHashJoinTransformerExec" ||
    trimmed == "ColumnarToRow" ||
    trimmed == "ColumnarToRowExec" ||
    // Check for "Scan parquet" pattern (with space)
    (trimmed.startsWith("Scan ") && trimmed.contains("parquet")) ||
    trimmed.startsWith("ArrowFile") ||
    trimmed.startsWith("Arrow") ||
    trimmed.contains("Columnar") ||
    trimmed.contains("Velox") ||
    trimmed.contains("CHColumnar") ||
    trimmed.endsWith("Transformer") ||
    trimmed.endsWith("TransformerExec") ||
    trimmed.contains("ScanTransformer") ||
    trimmed.contains("RegularHashAggregateExec") ||
    trimmed.contains("FlushableHashAggregateExec") ||
    trimmed.contains("HashAggregateTransformer") ||
    trimmed.contains("ShuffledHashJoinExec") ||
    trimmed.contains("ShuffledHashJoinTransformer") ||
    trimmed.contains("InputIterator") ||
    GLUTEN_PATTERN.findFirstIn(trimmed).isDefined

    if (isGluten) {
      logInfo(s"[GLUTEN-DEBUG] isGlutenNode: Recognized Gluten node: '$nodeName' " +
        s"(trimmed: '$trimmed')")
    }
    isGluten
  }

  /**
   * Replaces all occurrences in the input string that match Gluten patterns
   * with corresponding values from the glutenToSparkMapping map.
   *
   * @param inputStr the node name, potentially containing a Gluten identifier
   * @return the Spark node name
   */
  def mapGlutenToSpark(inputStr: String): String = {
    // Trim the input string to handle any whitespace issues
    val trimmed = inputStr.trim
    logInfo(s"[GLUTEN-DEBUG] mapGlutenToSpark: Mapping Gluten operator: '$inputStr' " +
      s"(trimmed: '$trimmed')")
    // Handle Exec suffix - remove it for matching and mapping
    // The mapped result should be the base Spark operator name (without Exec suffix)
    // because the parsers check for base names like "HashAggregate", not "HashAggregateExec"
    val hasExecSuffix = trimmed.endsWith("Exec")
    val nameWithoutExec = if (hasExecSuffix) trimmed.dropRight(4) else trimmed

    // Try to match full operator names first (sorted by length descending to match longer
    // names first). This ensures longer, more specific keys are matched before shorter ones.
    val sortedKeys = glutenToSparkMapping.keys.toSeq.sortBy(-_.length)
    val fullMatch = sortedKeys.find(key =>
      // Exact match (with or without Exec suffix) - check this first for accuracy
      trimmed == key ||
      nameWithoutExec == key ||
      // Exact match with Exec suffix removed matches key
      (hasExecSuffix && trimmed.dropRight(4) == key) ||
      // Starts with key followed by space (e.g., "ScanTransformer parquet")
      trimmed.startsWith(s"$key ") ||
      nameWithoutExec.startsWith(s"$key ") ||
      // Starts with key (e.g., "HashAggregateTransformer" or "HashAggregateTransformerExec")
      // But only if it's a complete word boundary (ends with Transformer/Exec or end of string)
      (trimmed.startsWith(key) && (
        trimmed.length == key.length ||
        trimmed.substring(key.length).matches("^(Exec|Transformer|ExecTransformer).*")
      )) ||
      (nameWithoutExec.startsWith(key) && (
        nameWithoutExec.length == key.length ||
        nameWithoutExec.substring(key.length).matches("^(Exec|Transformer|ExecTransformer).*")
      )) ||
      // Contains key followed by space (for embedded cases)
      trimmed.contains(s" $key ") ||
      nameWithoutExec.contains(s" $key ")
    )
    if (fullMatch.isDefined) {
      val mappedValue = glutenToSparkMapping(fullMatch.get)
      val key = fullMatch.get
      var result: String = ""

      // For operators with spaces (e.g., "ScanTransformer parquet"), replace only the operator part
      // and preserve the rest (e.g., "Scan parquet")
      if (trimmed.startsWith(s"$key ") || nameWithoutExec.startsWith(s"$key ")) {
        // Replace "ScanTransformer parquet" -> "Scan parquet"
        val baseStr = if (trimmed.startsWith(s"$key ")) trimmed else nameWithoutExec
        result = baseStr.replaceFirst(s"^$key ", s"$mappedValue ")
        // Don't add Exec suffix back - the mapped value is the base Spark operator name
      } else if (trimmed == key || nameWithoutExec == key) {
        // Exact match: "HashAggregateTransformer" -> "HashAggregate"
        // or "HashAggregateTransformerExec" -> "HashAggregate" (mapped value is base name)
        result = mappedValue
        // Don't add Exec suffix back - parsers expect base names
      } else if (trimmed.startsWith(key) || nameWithoutExec.startsWith(key)) {
        // Starts with key: "HashAggregateTransformerExec" -> "HashAggregate"
        val baseStr = if (trimmed.startsWith(key)) trimmed else nameWithoutExec
        result = baseStr.replaceFirst(s"^$key", mappedValue)
        // Remove Exec suffix if it exists after replacement
        if (result.endsWith("Exec") && !mappedValue.endsWith("Exec")) {
          result = result.dropRight(4)
        }
      } else {
        // Contains key: replace first occurrence
        val baseStr = if (trimmed.contains(key)) trimmed else nameWithoutExec
        result = baseStr.replaceFirst(key, mappedValue)
        // Remove Exec suffix if it exists after replacement
        if (result.endsWith("Exec") && !mappedValue.endsWith("Exec")) {
          result = result.dropRight(4)
        }
      }
      logInfo(s"[GLUTEN-DEBUG] mapGlutenToSpark: Mapped '$trimmed' -> '$result'")
      result
    } else {
        // Fall back to pattern-based replacement for common prefixes
        var result = nameWithoutExec
        // Handle "Scan parquet" pattern specifically
        if (trimmed.startsWith("Scan ") && trimmed.contains("parquet")) {
          // Map "Scan parquet" -> "Scan parquet" (already mapped, but ensure it's recognized)
          // Actually, if it's already "Scan parquet", it might be from a mapped node
          // Check if it matches "ScanTransformer parquet" pattern
          if (trimmed.contains("ScanTransformer")) {
            result = trimmed.replaceFirst("ScanTransformer", "Scan")
          } else {
            // Already "Scan parquet", return as-is
            result = trimmed
          }
        } else {
          // Remove common Gluten prefixes
          result = result.replaceFirst("^ArrowFile", "")
          result = result.replaceFirst("^Arrow", "")
          result = result.replaceFirst("^Columnar", "")
          result = result.replaceFirst("^Velox", "")
          result = result.replaceFirst("^CH", "")
          result = result.replaceFirst("Transformer$", "")
          // If no mapping found and no pattern matched, return as-is (without Exec suffix)
          if (result == nameWithoutExec) {
            result = nameWithoutExec
          }
        }
        logInfo(s"[GLUTEN-DEBUG] mapGlutenToSpark: Fallback mapping '$trimmed' -> " +
          s"'$result'")
        result
      }
  }
}
