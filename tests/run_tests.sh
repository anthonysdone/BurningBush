#!/bin/bash
# BurningBush Test Runner with Timeout Protection
# Prevents terminal freezes and captures all output

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TIMEOUT_SECONDS=5
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$TEST_DIR")"
OUTPUT_DIR="$TEST_DIR/output"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a single test with timeout
run_test_with_timeout() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .hc)
    local output_file="$OUTPUT_DIR/${test_name}_output.txt"
    local status_file="$OUTPUT_DIR/${test_name}_status.txt"
    
    echo -e "${BLUE}Running: $test_name${NC}"
    
    # Compile first (from project root so includes work)
    echo -e "  ${YELLOW}Compiling...${NC}"
    cd "$PROJECT_ROOT"
    if ! hcc "$test_file" -o "$OUTPUT_DIR/${test_name}_bin" 2>"$OUTPUT_DIR/${test_name}_compile.txt"; then
        echo -e "  ${RED}✗ COMPILATION FAILED${NC}"
        echo "COMPILE_FAIL" > "$status_file"
        cat "$OUTPUT_DIR/${test_name}_compile.txt"
        return 1
    fi
    echo -e "  ${GREEN}✓ Compiled${NC}"
    
    # Run with timeout
    echo -e "  ${YELLOW}Executing (${TIMEOUT_SECONDS}s timeout)...${NC}"
    
    # Run in background and capture PID
    "$OUTPUT_DIR/${test_name}_bin" > "$output_file" 2>&1 &
    local pid=$!
    
    # Wait with timeout
    local count=0
    while kill -0 $pid 2>/dev/null; do
        sleep 0.1
        count=$((count + 1))
        if [ $count -ge $((TIMEOUT_SECONDS * 10)) ]; then
            echo -e "  ${RED}✗ TIMEOUT (>${TIMEOUT_SECONDS}s)${NC}"
            kill -9 $pid 2>/dev/null || true
            echo "TIMEOUT" > "$status_file"
            echo -e "\n${YELLOW}Partial output:${NC}"
            cat "$output_file"
            return 1
        fi
    done
    
    # Check exit status
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "  ${GREEN}✓ PASSED${NC}"
        echo "PASS" > "$status_file"
        cat "$output_file"
        return 0
    else
        echo -e "  ${RED}✗ FAILED (exit code: $exit_code)${NC}"
        echo "FAIL" > "$status_file"
        cat "$output_file"
        return 1
    fi
}

# Function to run all tests in a directory
run_test_suite() {
    local suite_dir="$1"
    local suite_name=$(basename "$suite_dir")
    
    echo ""
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Test Suite: $suite_name${NC}"
    echo -e "${BLUE}================================${NC}"
    
    local total=0
    local passed=0
    local failed=0
    local timeout=0
    local compile_fail=0
    
    # Find all .hc test files
    for test_file in "$suite_dir"/*.hc; do
        if [ -f "$test_file" ]; then
            total=$((total + 1))
            
            if run_test_with_timeout "$test_file"; then
                passed=$((passed + 1))
            else
                local test_name=$(basename "$test_file" .hc)
                local status=$(cat "$OUTPUT_DIR/${test_name}_status.txt" 2>/dev/null || echo "UNKNOWN")
                
                case "$status" in
                    TIMEOUT)
                        timeout=$((timeout + 1))
                        ;;
                    COMPILE_FAIL)
                        compile_fail=$((compile_fail + 1))
                        ;;
                    *)
                        failed=$((failed + 1))
                        ;;
                esac
            fi
            echo ""
        fi
    done
    
    # Summary
    echo -e "${BLUE}--------------------------------${NC}"
    echo -e "Suite: $suite_name"
    echo -e "Total:   $total"
    echo -e "${GREEN}Passed:  $passed${NC}"
    echo -e "${RED}Failed:  $failed${NC}"
    echo -e "${YELLOW}Timeout: $timeout${NC}"
    echo -e "${RED}Compile: $compile_fail${NC}"
    echo -e "${BLUE}--------------------------------${NC}"
    
    return $((failed + timeout + compile_fail))
}

# Main execution
main() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}BurningBush Test Runner${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo -e "Project Root: $PROJECT_ROOT"
    echo -e "Output Dir:   $OUTPUT_DIR"
    echo -e "Timeout:      ${TIMEOUT_SECONDS}s"
    
    # Clean previous outputs
    rm -f "$OUTPUT_DIR"/*.txt "$OUTPUT_DIR"/*_bin
    
    local total_failures=0
    
    # Run specific test if provided as argument
    if [ $# -gt 0 ]; then
        if [ -f "$1" ]; then
            run_test_with_timeout "$1"
            exit $?
        else
            echo -e "${RED}Error: Test file not found: $1${NC}"
            exit 1
        fi
    fi
    
    # Run all test suites
    for suite_dir in "$TEST_DIR"/*/; do
        if [ -d "$suite_dir" ]; then
            run_test_suite "$suite_dir"
            total_failures=$((total_failures + $?))
        fi
    done
    
    # Final summary
    echo ""
    echo -e "${BLUE}======================================${NC}"
    if [ $total_failures -eq 0 ]; then
        echo -e "${GREEN}ALL TESTS PASSED!${NC}"
        exit 0
    else
        echo -e "${RED}SOME TESTS FAILED (total issues: $total_failures)${NC}"
        exit 1
    fi
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n${YELLOW}Test run interrupted${NC}"; exit 130' INT TERM

main "$@"
